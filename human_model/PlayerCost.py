import numpy as np

class PlayerCost:
    def __init__(self):
        self.costs = []
        self.weights = []
        self.args = []

    def add_cost(self, cost, arg, weight):
        self.costs.append(cost)
        self.args.append(arg)
        self.weights.append(weight)
    
    def cost(self, x, u):
        total_cost = 0.0

        for i  in range(len(self.costs)):

            current_cost = self.costs[i]

            if self.args[i] == 'x':
                current_term = self.weights[i]*current_cost.calc_r(x)
            elif self.args[i] == 'u':
                current_term = self.weights[i]*current_cost.calc_r(u)
            elif self.args[i] == 'xu':
                current_term = self.weights[i]*current_cost.calc_r(x,u)
            else:
                Warning("Invalid args")

            total_cost = total_cost + current_term
        
        return total_cost

    def quadraticize(self, x, u):
        total_cost = self.cost(x,u)

        dldx = np.zeros((len(x), 1))
        Hx = np.zeros( (len(x), len(x)) )
        Hu = np.zeros( (len(u), len(u)) )

        for i in range(len(self.costs)):
            current_cost = self.costs[i]
            dldx = dldx + current_cost.calc_dldx(x)
            Hx = Hx + current_cost.calc_Hx(x)
            Hu = Hu + current_cost.calc_Hu(u)
        
        return total_cost, dldx, Hx, Hu




