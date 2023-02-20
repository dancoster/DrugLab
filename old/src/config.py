import os

class Config:
    
    def __init__(self, args, BASE_DIR, DATA):
        self.config = dict()
    
    def get(self, arg):
        pass
    
    def add(self, arg):
        config['data_path'] = get_arg_val('--data', args)
        if config['data_path'] is None:
            config['data_path'] = os.path.join(DATA, 'mimiciii', '1.4')
        pass
    
    def get_arg_val(self, arg, k=1):
        if k==1:
            if arg in self.args:
                ind = self.args.index(arg)
                val = self.args[ind+1]
                return val
        if k>1:
            if arg in self.args:
                ind = self.args.index(arg)
                val = []
                for i in range(k):
                    val.append(self.args[ind+i+1])
                return val
        return None
    
    def config_args(args, BASE_DIR, DATA):

        config = dict()

        config['data_path'] = get_arg_val('--data', args)
        if config['data_path'] is None:
            config['data_path'] = os.path.join(DATA, 'mimiciii', '1.4')

        config['results_path'] = get_arg_val('--results', args)
        if config['results_path'] is None:
            config['results_path'] = BASE_DIR+'/results'

        config['n_meds'] = get_arg_val('--meds', args)
        if config['n_meds'] is None:
            config['n_meds'] = 50
        config['n_meds'] = int(config['n_meds'])

        config['n_sub_pairs'] = get_arg_val('--sub-pairs', args)
        if config['n_sub_pairs'] is None:
            config['n_sub_pairs'] = 50
        config['n_sub_pairs'] = int(config['n_sub_pairs'])

        config['table'] = get_arg_val('--table', args)
        if config['table'] is None:
            config['table'] = 'inputevents'

        config['window'] = get_arg_val('--window', args, k=2)
        if config['window'] is None:
            config['window'] = (1,24)
        else:
            config['window'] = (int(config['window'][0]), int(config['window'][1]))
        
        config['visualize'] = get_arg_val('--visualize', args, k=2)
        if config['visualize'] is None:
            config['visualize'] = ('Insulin - Regular', 'Glucose')
        else:
            config['visualize'] = tuple(config['visualize'])

        return config
