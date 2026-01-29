import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HyperNetwork(nn.Module):
    def __init__(self, embedding_model, embedding_output_size, num_weights, num_biases):
        super().__init__()
        self.embedding_model = embedding_model
        self.embedding_model_params = [param for param in embedding_model.parameters() if param.requires_grad]
        self.num_weights = num_weights
        self.weights_gen = nn.Linear(in_features=embedding_output_size, out_features=embedding_output_size)
        self.parameters_generators_input_size = embedding_output_size

    def calc_variance4init(self, args, main_net_in_size, train_dataloader, hyper_input_type,
                           embd_vars=False, main_net_relu=True, main_net_biasses=True, var_hypernet_input=None):
    # initialize the weights of the weights geneerator
        variances = []
        with torch.no_grad():
            for iteration, data in enumerate(train_dataloader): #enumerate(tqdm(test_loader, leave=False)):
                bags, table, label = data["bags"], data["table"], data["label"]
                bags, table, label  = bags.to(args.device), table.to(args.device), label.to(args.device)

                values = self.embedding_model(table)
                for v in values:
                    variances += [np.array(v.view(-1).detach().cpu()).var()]    
                    
        var_hypernet_input = np.mean(variances)
        if var_hypernet_input == 0:
            var_hypernet_input = 1

        # calculate the needed variance
        dk = self.parameters_generators_input_size  # both dk and dl
        dj = main_net_in_size
        var_weights_generator = (2 ** main_net_relu) / ((2 ** main_net_biasses) * dj * dk * var_hypernet_input)
        var_biasses_generator = None
        return var_weights_generator, var_biasses_generator

    def variance_uniform_init(self, var_weights_generator, var_biasses_generator):
        # initialize the weights of the weights geneerator
        # according to PRINCIPLED WEIGHT INITIALIZATION FOR HYPERNETWORKS

        # apply the initialization
        ws_init = np.sqrt(3 * var_weights_generator)
        nn.init.uniform_(self.weights_gen.weight, -ws_init, ws_init)

        # init the biasses of the weights generators with 0
        nn.init.constant_(self.weights_gen.bias, 0)

    def initialize_parameters(self, args, weights_init_method, fan_in, hyper_input_type,
                              for_conv=False, train_loader=None, GPU=None, var_hypernet_input=None):
        if weights_init_method == "input_variance":
            print("input_variance weights initialization")
            var_w, var_b = self.calc_variance4init(args, fan_in, train_loader, hyper_input_type, embd_vars=False,
                                                   var_hypernet_input=var_hypernet_input)
            self.variance_uniform_init(var_w, var_b)
        elif weights_init_method == "embedding_variance":
            print("embedding_variance weights initialization")
            var_w, var_b = self.calc_variance4init(args, fan_in, train_loader, hyper_input_type, embd_vars=True)
            self.variance_uniform_init(var_w, var_b)
        else:
            raise ValueError("HyperNetwork initialization type not implemented!")

    def freeze_embedding_model(self):
        for param in self.embedding_model_params:
            param.requires_grad = False

    def unfreeze_embedding_model(self):
        for param in self.embedding_model_params:
            param.requires_grad = True

    def forward(self, x):
        emb_out = self.embedding_model(x)
        weights = self.weights_gen(emb_out)
        biases = None
        return weights, biases


class HyperLinearLayer_token(nn.Module):
    def __init__(self, args, in_features, out_features, embedding_model, embedding_output_size,
                 weights_init_method="embedding_variance", train_loader=None, hyper_input_type=None, GPU=None, var_hypernet_input=None):
        super().__init__()
        num_weights = in_features * out_features
        num_biases = out_features
        self.hyper_net = HyperNetwork(embedding_model, embedding_output_size, num_weights, num_biases)

        self.num_out_features = out_features
        self.weights_shape = (out_features, in_features)

        # initialize the weights of the layer if there is weights_init_method value
        if not (weights_init_method is None):
            self.hyper_net.initialize_parameters(args, weights_init_method, in_features, hyper_input_type,
                                                 for_conv=False, train_loader=train_loader, GPU=GPU,
                                                 var_hypernet_input=var_hypernet_input)

    def forward(self, x):
        x = self.hyper_net(x)  # creates #batch_size sets of parameters for the linear operation
        return x