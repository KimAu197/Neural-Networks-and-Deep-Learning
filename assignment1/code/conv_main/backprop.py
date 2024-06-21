
from builtins import range
from builtins import object
from original import optim
import numpy as np
import pickle as pickle

class Backprop(object):
    def __init__(self, model, data, max_iteration, checkpoint = None, **kwargs):

        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.max_iteration = max_iteration
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.print_every = kwargs.pop("print_every", 10)
        self.checkpoint_name = checkpoint
        self.update_rule = getattr(optim, self.update_rule)
        
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """

        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = [0]


        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):


        loss, grads = self.model.loss(self.X_train, self.y_train)
        self.loss_history.append(loss)


        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
            
    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
            "mode_params": self.model.params
        }

        folder_name = "checkpoint"
        filename = "%s.pkl" % (self.checkpoint_name)
        print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)
        
    def check_accuracy(self, X, y):

        y_pred = []
        scores = self.model.loss(X) 
        y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):

        num_iterations = self.max_iteration

        for t in range(num_iterations):
            self._step()
            # every 300 iteration decay
            if num_iterations % 300 == 0:
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay


            first_it = t == 0
            last_it = t == num_iterations - 1
            
            if first_it or last_it or t % self.print_every == 0:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val
                )
                # early stop
                if val_acc <= self.val_acc_history[-1]:
                    break
                else:
                    self.train_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)
                    
                    print("(training iteration %d / %d) train acc: %f; val_acc: %f"
                            % (t, num_iterations, train_acc, val_acc))
                    self._save_checkpoint()

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        self.model.params = self.best_params
