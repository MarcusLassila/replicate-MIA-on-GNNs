import attacks
import datasetup
import hypertuner
import evaluation
import trainer
import utils

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from collections import defaultdict
from pathlib import Path
import copy
import time

class MembershipInferenceAudit:

    def __init__(self, config):
        config = utils.Config(config)
        self.device = torch.device(config.device)
        self.dataset = datasetup.parse_dataset(root=config.datadir, name=config.dataset, max_num_nodes=config.max_num_nodes)
        self.dataset.to(self.device)
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.shadow_models = None
        print(utils.graph_info(self.dataset))
        if config.hyperparam_search:
            val_frac = config.val_frac or config.train_frac
            _ = datasetup.random_remasked_graph(self.dataset, train_frac=config.train_frac, val_frac=val_frac, mutate=True)
            opt_hyperparams = hypertuner.grid_search(
                param_grid=config.hyperparam_grid,
                dataset=self.dataset,
                model_type=config.model,
                optimizer=config.optimizer,
                inductive_split=config.inductive_split,
                minimum_average_gen_gap=config.minimum_average_gen_gap,
            )
            print(f'Hyperparameter search results: {opt_hyperparams}')
            Path(f"results/hyperparams/{config.dataset}").mkdir(parents=True, exist_ok=True)
            log_info = f'dataset: {config.dataset}\nmodel: {config.model}\nnum_nodes: {self.dataset.num_nodes}\n' + '\n'.join(f'{k}: {v}' for k, v in opt_hyperparams.items())
            with open(f"results/hyperparams/{config.dataset}/{config.dataset}_{config.model}_{self.dataset.num_nodes}.txt", "w") as f:
                f.write(log_info)
        self.config = config

    def train_target_model(self, dataset):
        config = self.config
        target_model = utils.fresh_model(
            model_type=config.model,
            num_features=dataset.num_features,
            hidden_dims=config.hidden_dim,
            num_classes=dataset.num_classes,
            dropout=config.dropout,
        )
        train_config = trainer.TrainConfig(
            criterion=self.criterion,
            device=self.device,
            epochs=config.epochs,
            early_stopping=config.early_stopping,
            loss_fn=self.loss_fn,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        print(f'Training a {config.model} target model on {config.dataset}...')
        _ = trainer.train_gnn(
            model=target_model,
            dataset=dataset,
            config=train_config,
            inductive_split=config.inductive_split
        )
        evaluation.evaluate_graph_training(
            model=target_model,
            dataset=dataset,
            criterion=train_config.criterion,
            inductive_inference=config.inductive_split,
        )
        return target_model

    def tune_attack_hyperparams(self, n_cross_val=8):
        if n_cross_val == 0:
            return
        assert n_cross_val <= self.config.num_shadow_models

        def run_optuna(attack, attack_dict, n_cross_val):
            attack_config = utils.Config(attack_dict)
            for cv_i in range(n_cross_val):
                if cv_i % 2 == 0:
                    i_a, i_b = cv_i, cv_i + 2
                else:
                    i_a, i_b = cv_i - 1, cv_i + 1
                simul_target, simul_target_train_mask = self.shadow_models[cv_i]
                simul_graph = datasetup.remasked_graph(self.dataset, simul_target_train_mask)
                # Do not include the shadow model that is trained on the node complement of the simulated target
                # Otherwise each node is not included in the training set of half the models
                simul_shadow_models = self.shadow_models[:i_a] + self.shadow_models[i_b:]
                n_trials = 100
                match attack:
                    case 'online-base' | 'offline-base':
                        hyperparam_name = 'threshold_scale_factor'
                        if hasattr(attack_config, hyperparam_name):
                            return ''
                        print(f'Tuning threshold scale factor for {attack.replace('-', ' ')} using optuna')
                        simul_attacker = attacks.BASE(
                            target_model=simul_target,
                            graph=simul_graph,
                            loss_fn=self.loss_fn,
                            config=attack_config,
                            shadow_models=simul_shadow_models,
                        )
                    case 'offline-rmia':
                        hyperparam_name = 'interp_param'
                        if hasattr(attack_config, hyperparam_name):
                            return ''
                        print('Tuning interpolation parameter for offline rmia using optuna')
                        simul_attacker = attacks.RMIA(
                            target_model=simul_target,
                            graph=simul_graph,
                            loss_fn=self.loss_fn,
                            config=attack_config,
                            shadow_models=simul_shadow_models,
                        )
                    case _:
                        continue
                hyperparam_value = hypertuner.optuna_hyperparam_tuner(
                    simul_attacker,
                    hyperparam_name,
                    n_trials=n_trials,
                    execute_silently=True,
                )
                if hyperparam_name in attack_dict:
                    attack_dict[hyperparam_name].append(hyperparam_value)
                else:
                    attack_dict[hyperparam_name] = [hyperparam_value]
            return hyperparam_name

        for attack_dict in self.config.attacks.values():
            mode = 'offline' if attack_dict['offline'] else 'online'
            attack = mode + '-' + attack_dict['attack']
            if attack not in ('online-base', 'offline-base', 'offline-rmia'):
                continue
            tuned_hyperparam = run_optuna(attack, attack_dict, n_cross_val)
            if tuned_hyperparam:
                values = np.array(attack_dict[tuned_hyperparam])
                assert values.shape[0] == n_cross_val, "Should have exactly one value per cross validation"
                attack_dict[tuned_hyperparam] = values.mean()
                std = values.std() if n_cross_val > 1 else 0.0
                print(f'{tuned_hyperparam}: {attack_dict[tuned_hyperparam]} +- {std}')

    def get_attacker(self, attack_dict, target_model):
        '''
        Return an instance of the attack class specified by config.attack
        '''
        attack_config = utils.Config(attack_dict)
        pretrained_shadow_models = self.shadow_models
        if self.config.edge_noise_level > 0.0:
            graph = self.dataset.clone()
            graph.edge_index = datasetup.noisy_edge_index(self.dataset.edge_index, self.dataset.num_nodes, noise_lvl=self.config.edge_noise_level)
        else:
            graph = self.dataset
        match attack_config.attack:
            case "g-base":
                attacker = attacks.G_BASE(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "base":
                attacker = attacks.BASE(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "laplace-base":
                attacker = attacks.LaplaceBASE(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "b-base":
                attacker = attacks.B_BASE(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "bg-base":
                attacker = attacks.BG_BASE(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "s-base":
                attacker = attacks.S_BASE(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "sg-base":
                attacker = attacks.SG_BASE(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "confidence":
                attacker = attacks.ConfidenceAttack(
                    target_model=target_model,
                    graph=graph,
                    config=attack_config,
                )
            case "bmia":
                attacker = attacks.BMIA(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                )
            case "lira":
                attacker = attacks.LiRA(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "rmia":
                attacker = attacks.RMIA(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case "mlp-attack":
                attacker = attacks.MLPAttack(
                    target_model=target_model,
                    graph=graph,
                    loss_fn=self.loss_fn,
                    config=attack_config,
                    shadow_models=pretrained_shadow_models,
                )
            case _:
                raise AttributeError(f"No attack named {attack_config.attack}")
        return attacker

    def get_target_nodes(self):
        config = self.config
        assert 0.0 <= config.frac_target_nodes <= 1.0
        assert not torch.any(self.dataset.train_mask & self.dataset.test_mask)
        num_target_nodes = int(config.frac_target_nodes * self.dataset.num_nodes)
        # Make sure the number of targets is even so there can be an equal amount of members and non-members
        if num_target_nodes % 2 == 1:
            num_target_nodes -= 1
        train_nodes = self.dataset.train_mask.long()
        test_nodes = self.dataset.test_mask.long()
        positives = train_nodes.nonzero().squeeze()
        negatives = test_nodes.nonzero().squeeze()
        num_target_nodes = min(num_target_nodes, 2 * positives.shape[0])
        perm_mask = torch.randperm(positives.shape[0])
        positives = positives[perm_mask][:num_target_nodes // 2]
        perm_mask = torch.randperm(negatives.shape[0])
        negatives = negatives[perm_mask][:num_target_nodes // 2]
        perm_mask = torch.randperm(num_target_nodes)
        target_node_index = torch.concat((positives, negatives))[perm_mask]
        assert target_node_index.shape == (num_target_nodes,)
        return target_node_index

    def load_model(self, path, return_target_node_index=False):
        '''
        Assumes a path to a pickle file, storing a dict with:
        1. model_state_dict
        2. train_mask

        Return tuple (model, train_mask).
        '''
        model_state = torch.load(path, map_location=self.device)
        assert model_state['num_features'] == self.dataset.num_features
        assert model_state['num_classes'] == self.dataset.num_classes
        model = utils.fresh_model(
            model_type=model_state['model_type'],
            num_features=model_state['num_features'],
            hidden_dims=model_state['hidden_dim'],
            num_classes=model_state['num_classes'],
            dropout=model_state['dropout'],
        ).to(self.device)
        model.load_state_dict(model_state['model_state_dict'])
        model.eval()
        if return_target_node_index:
            return model, model_state['train_mask'], model_state['target_node_index']
        else:
            return model, model_state['train_mask']

    def save_model(self, path, model, train_mask, target_node_index=None):
        if os.path.exists(path):
            raise FileExistsError(f'File {path} already exists.')
        config = self.config
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_state = {
            'model_state_dict': model.state_dict(),
            'train_mask': train_mask,
            'model_type': config.model,
            'num_features': self.dataset.num_features,
            'hidden_dim': config.hidden_dim,
            'num_classes': self.dataset.num_classes,
            'dropout': config.dropout,
            'target_node_index': target_node_index,
        }
        torch.save(model_state, path)

    def parse_stats(self, stats):
        config = self.config
        frames = []
        for attack in config.attacks.keys():
            table = defaultdict(list)
            for key, value in stats[attack].items():
                if key not in ('TPR', 'FPR'):
                    table[f'{key}'].append(utils.stat_repr(value))
            frames.append(pd.DataFrame(table, index=[config.name + '_' + attack]))
        return pd.concat(frames)

    def run_audit(self):
        config = self.config
        stats = defaultdict(lambda: defaultdict(list))
        if config.pretrain_shadow_models:
            t0 = time.time()
            self.shadow_models = trainer.train_shadow_models(self.dataset, self.loss_fn, config)
            t1 = time.time()
            print(f'{config.num_shadow_models} shadow model trained in {t1 - t0:.2f} seconds')
            for i, (shadow_model, shadow_train_mask) in enumerate(self.shadow_models):
                self.save_model(
                    path=f'{config.modeldir}/{config.dataset}-{config.model}/shadow-model-{i}.pth',
                    model=shadow_model,
                    train_mask=shadow_train_mask,
                )

        for i_audit in range(config.target_index_start, config.target_index_start + config.num_audits):
            i_audit_centered = i_audit - config.target_index_start
            print(f'Running audit {i_audit_centered + 1}/{config.num_audits}')

            # Load shadow models
            self.shadow_models = []
            if config.shadow_model_path:
                shadow_model_path = config.shadow_model_path
            else:
                shadow_model_path = f'{config.modeldir}/{config.dataset}-{config.model}/shadow-model'
            if config.fixed_shadow_models or config.pretrain_shadow_models:
                shadow_model_index_range = [*range(config.num_shadow_models)]
            else:
                shadow_model_index_range = [*range(i_audit * config.num_shadow_models, (i_audit + 1) * config.num_shadow_models)]
            print(f'Loading shadow models {shadow_model_path}-k.pth for {shadow_model_index_range[0]} <= k <= {shadow_model_index_range[-1]}')
            for shadow_model_index in shadow_model_index_range:
                path = f'{shadow_model_path}-{shadow_model_index}.pth'
                shadow_model, shadow_train_mask = self.load_model(path)
                self.shadow_models.append((shadow_model, shadow_train_mask))

            # Tune attack specific hyperparameters with optuna
            if i_audit == config.target_index_start:
                self.tune_attack_hyperparams(n_cross_val=config.hyperparam_cross_vals)

            # Load target model
            if config.target_model_path:
                path = f'{config.target_model_path}-{i_audit}.pth'
                print(f'Loading target model: {path}')
                target_model, target_train_mask, target_node_index = self.load_model(path, return_target_node_index=True)
                _ = datasetup.remasked_graph(self.dataset, target_train_mask, mutate=True)
                ground_truth = target_train_mask.long()[target_node_index]
            else:
                _ = datasetup.random_remasked_graph(self.dataset, train_frac=config.train_frac, val_frac=config.val_frac, mutate=True)
                assert not torch.any(self.dataset.val_mask), "Validation mask not fully supported"
                assert not torch.any(self.dataset.train_mask & self.dataset.test_mask)
                assert torch.all(self.dataset.train_mask | self.dataset.test_mask)
                target_node_index = self.get_target_nodes()
                target_model = self.train_target_model(self.dataset)
                target_model.eval()
                ground_truth = self.dataset.train_mask.long()[target_node_index]
                self.save_model(
                    path=f'{config.modeldir}/{config.dataset}-{config.model}/target-model-{str(config.frac_target_nodes).replace(".", "")}-{config.optimizer}-{i_audit}.pth',
                    model=target_model,
                    train_mask=self.dataset.train_mask,
                    target_node_index=target_node_index,
                )
            target_scores = {
                'train_acc': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=self.dataset,
                    mask=self.dataset.train_mask,
                    criterion=self.criterion,
                    inductive_inference=config.inductive_split,
                ),
                'test_acc': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=self.dataset,
                    mask=self.dataset.test_mask,
                    criterion=self.criterion,
                    inductive_inference=config.inductive_split,
                ),
            }
            for attack, attack_dict in config.attacks.items():
                attacker = self.get_attacker(attack_dict, target_model)
                stats[attack]['train_acc'].append(target_scores['train_acc'])
                stats[attack]['test_acc'].append(target_scores['test_acc'])
                t0 = time.time()
                preds = attacker.run_attack(target_node_index=target_node_index)
                t1 = time.time()
                stats[attack]['time'].append(t1 - t0)
                metrics = evaluation.evaluate_binary_classification(preds, ground_truth, config.target_fpr, target_node_index, self.dataset)
                fpr, tpr = metrics['ROC']
                stats[attack]['FPR'].append(fpr)
                stats[attack]['TPR'].append(tpr)
                stats[attack]['AUC'].append(metrics['AUC'])
                for t_fpr, t_tpr, threshold in zip(config.target_fpr, metrics['TPR@FPR'], metrics['threshold@FPR']):
                    stats[attack][f'TPR@{t_fpr}FPR'].append(t_tpr)
                    if config.save_thresholds:
                        stats[attack][f'threshold@{t_fpr}FPR'].append(threshold)

        stat_df = self.parse_stats(stats)
        stats = utils.nestled_defaultdict_to_dict(stats)
        return stat_df, stats

def add_attack_parameters(params):
    ''' Add target values as default values to attack config parameters. '''
    properties = [
        'model', 'epochs', 'hidden_dim', 'lr', 'weight_decay', 'optimizer', 'dropout',
        'inductive_split', 'device', 'early_stopping', 'train_frac', 'val_frac',
        'num_processes', 'num_shadow_models', 'offline',
    ]
    for attack_params in params['attacks'].values():
        for prop in properties:
            if prop not in attack_params:
                attack_params[prop] = params[prop]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run(config):
    set_seed(config['seed'])
    if config['hyperparam_search']:
        MembershipInferenceAudit(config)
    else:
        add_attack_parameters(config)
        mie = MembershipInferenceAudit(config)
        return mie.run_audit()
