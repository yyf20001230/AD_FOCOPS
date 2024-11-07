import os
import pickle
import torch
from utils.misc import println
from torch.utils.tensorboard import SummaryWriter

class Logger:

    def __init__(self, hyperparams):

        self.log_data = {'time': 0,
                         'MinR': [],
                         'MaxR': [],
                         'AvgR': [],
                         'MinC': [],
                         'MaxC': [],
                         'AvgC': [],
                         'nu': [],
                         'running_stat': None}

        self.models = {'iter': None,
                       'policy_params': None,
                       'value_params': None,
                       'cvalue_params': None,
                       'pi_optimizer': None,
                       'vf_optimizer': None,
                       'cvf_optimizer': None,
                       'pi_loss': None,
                       'vf_loss': None,
                       'cvf_loss': None}

        self.hyperparams = hyperparams
        self.iter = 0



    def update(self, key, value):
        if type(self.log_data[key]) is list:
            self.log_data[key].append(value)
        else:
            self.log_data[key] = value

    def save_model(self, component, params):
        self.models[component] = params


    def dump(self):
        batch_size = self.hyperparams['batch_size']
        total_samples = (self.iter + 1) * batch_size
        
        # Get the run name components
        env_id = self.hyperparams['env_id']
        constraint = self.hyperparams['constraint']
        seed = self.hyperparams['seed']
        delay_steps = self.hyperparams['delay_steps']
        envname = env_id.partition(':')[-1] if ':' in env_id else env_id
        
        # Create a unique run name
        run_name = f"focops_{constraint}_{envname}_seed{seed}_delay{delay_steps}"
        
        # Create TensorBoard writer if it doesn't exist
        if not hasattr(self, 'writer'):
            log_dir = os.path.join("runs", run_name)
            self.writer = SummaryWriter(log_dir)
        
        # Log metrics to TensorBoard
        self.writer.add_scalar('Training/Time', self.log_data['time'], self.iter)
        
        # Rewards
        self.writer.add_scalar('Rewards/Min', self.log_data['MinR'][-1], self.iter)
        self.writer.add_scalar('Rewards/Max', self.log_data['MaxR'][-1], self.iter)
        self.writer.add_scalar('Rewards/Average', self.log_data['AvgR'][-1], self.iter)
        
        # Constraints
        self.writer.add_scalar('Constraints/Min', self.log_data['MinC'][-1], self.iter)
        self.writer.add_scalar('Constraints/Max', self.log_data['MaxC'][-1], self.iter)
        self.writer.add_scalar('Constraints/Average', self.log_data['AvgC'][-1], self.iter)
        
        # Nu parameter
        self.writer.add_scalar('Parameters/Nu', self.log_data['nu'][-1], self.iter)
        
        # Print results (keeping the console output for convenience)
        println('Results for Iteration:', self.iter + 1)
        println('Number of Samples:', total_samples)
        println('Time: {:.2f}'.format(self.log_data['time']))
        println('MinR: {:.2f}| MaxR: {:.2f}| AvgR: {:.2f}'.format(
            self.log_data['MinR'][-1],
            self.log_data['MaxR'][-1],
            self.log_data['AvgR'][-1]
        ))
        println('MinC: {:.2f}| MaxC: {:.2f}| AvgC: {:.2f}'.format(
            self.log_data['MinC'][-1],
            self.log_data['MaxC'][-1],
            self.log_data['AvgC'][-1]
        ))
        println('Nu: {:.3f}'.format(self.log_data['nu'][-1]))
        println('--------------------------------------------------------------------')
        
        # Save models (keeping model saving functionality)
        directory = 'focops_results'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_filename = os.path.join(directory, f"{run_name}_model.pth")
        torch.save(self.models, model_filename)
        
        # Advance iteration
        self.iter += 1
        
        # # Ensure we flush the TensorBoard writer
        self.writer.flush()

    def close(self):
        """
        Method to properly close the TensorBoard writer
        Should be called when training is complete
        """
        if hasattr(self, 'writer'):
            self.writer.close()
