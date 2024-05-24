import torch
import numpy as np
import warnings
from scipy.special import binom

class smooth_max(object):
    def __init__(self, T=0.1):
        self.T = T

    def __call__(self, x):
        return self.T * torch.log((1.0 / len(x))
                                  * torch.sum(torch.exp(x / self.T)))

def S_(x, f):
    return np.power(-1, np.floor(4.0 * f * x))

def in_Z(x):
    return (int(x) == x) and (x >= 0)


class ddKS(object):
    def __init__(self, soft=False, T=0.1, method='all', n_test_points=10,
                 pts=None, norm=False, oneway=False):
        if soft:
            self.max = smooth_max(T=T)
            self.ge = self.softge
        else:
            self.max = torch.max
            self.ge = self.hardge
        self.method = method
        self.n_test_points = n_test_points
        self.pts = pts
        self.norm = norm
        self.oneway = oneway

    def __call__(self, pred, true):
        '''
        Takes in two distributions and returns ddks distance
        For child classes define setup/calcD
        :param pred: [N1 x d] tensor
        :param true: [N2 x d] tensor
        :return:
        '''
        self.pred = pred
        self.true = true
        #Enforce N x d and d1=d2
        if len(pred.shape) < 2 or len(true.shape) < 2:
            warnings.warn(f'Error Pred or True is missing a dimension')
        if pred.shape[1] != true.shape[1]:
            warnings.warn(f'Dimension Mismatch between pred/true: Shapes should be [n1,d], [n2,d]')

        self.setup(pred,true)

        D = self.calcD(pred, true)


        return D
    def setup(self,pred,true):
        self.getQU(pred,true)

    def M(self, sample, test_points):
        if sample.shape[1] != 3:
            get_ort = self.get_orthants
        else:
            get_ort = self.get_octants
        _M = get_ort(sample, test_points)
        return _M

    def calcD(self, pred, true):
        if pred.shape[1] != 3:
            get_ort = self.get_orthants
        else:
            get_ort = self.get_octants
        os_pp = get_ort(pred, self.Q)
        os_pt = get_ort(true, self.Q)
        D1 = self.max((os_pp - os_pt).abs())
        if self.oneway:
            D = D1
        else:
            os_tt = get_ort(true, self.U)
            os_tp = get_ort(pred, self.U)
            D2 = self.max((os_tt - os_tp).abs())
            D = max(D1,D2)
        if self.norm:
            D = D / float(pred.shape[0])
        return D
    ###
    # Setup Functions
    ###
    def getQU(self, pred, true):
        # Uses self.method to choose to use all points to split space or subsample
        # or use grid
        if self.method == 'all':
            Q = pred
            U = true
        elif self.method == 'subsample':
            idx = np.random.choice(np.arange(np.min([pred.shape[0], true.shape[0]])), size=self.n_test_points)
            Q = pred[idx, ...]
            U = true[idx, ...]
        elif self.method == 'linear':
            if self.pts is None:
                Q = torch.empty((self.n_test_points, pred.shape[1]))
                U = torch.empty((self.n_test_points, true.shape[1]))
                for dim in range(pred.shape[1]):
                    Q[:, dim] = torch.linspace(pred[:, dim].min(), pred[:, dim].max(), self.n_test_points)
                for dim in range(true.shape[1]):
                    U[:, dim] = torch.linspace(true[:, dim].min(), true[:, dim].max(), self.n_test_points)
        self.Q = Q
        self.U = U
        return
    ###
    # calcD functions
    ###
    def get_orthants(self, x, points):
        '''
        n-Dimensional version of get_octants (probably faster to run 3-D using get_orthants)
        :param x: Either pred/true used the samples being placed into orthants
        :param points: The points being used to create orthants i.e Q/U if self.method='all' Q/U = pred/true
        :return: row-Normalized occupancy matrix - each element corresponds to the occupancy % in an orthant
        '''
        N = x.shape[0]
        d = points.shape[1]
        # shape our input and test points into the right shape (N, 3, 1)
        x = x.unsqueeze(-1)
        points = points.unsqueeze(-1)
        # repeat each input point in the dataset across the third dimension
        x = x.repeat((1, 1, points.shape[0]))
        # repeate each test in the dataset across the first dimension
        comp_x = points.repeat((1, 1, x.shape[0]))
        comp_x = comp_x.permute((2, 1, 0))
        # now compare the input points and comparison points to see how many
        # are bigger and smaller
        x = self.ge(x, comp_x)
        orthants = []
        orthant_matrix = self.get_orthant_matrix(d)
        for i in range(int(np.power(2, d))):
            membership = 1.0
            for j in range(d):
                membership *= (float(orthant_matrix[i, j] < 0) + orthant_matrix[i, j] * x[:, j, :]).abs()
            orthant = torch.sum(membership, dim=0).float() / N
            orthants.append(orthant)
        return torch.stack(orthants, dim=1)

    def get_orthant_matrix(self, d):
        n_orthants = int(np.power(2, d))
        x = np.empty((n_orthants, d))
        for i in range(n_orthants):
            for j in range(d):
                x[i, j] = S_(i, np.power(2.0, -j - 2))
        return x

    def get_octants(self, x, points):
        N = x.shape[0]
        # shape our input and test points into the right shape (N, 3, 1)
        x = x.unsqueeze(-1)
        points = points.unsqueeze(-1)
        # repeat each input point in the dataset across the third dimension
        x = x.repeat((1, 1, points.shape[0]))
        # repeate each test in the dataset across the first dimension
        comp_x = points.repeat((1, 1, x.shape[0]))
        comp_x = comp_x.permute((2, 1, 0))
        # now compare the input points and comparison points to see how many
        # are bigger and smaller
        x = self.ge(x, comp_x)
        nx = (1 - torch.clone(x)).abs()
        # now use the comparisoned points to construct each octant (& is logical and)
        o1 = torch.sum(x[:, 0, :] * x[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o2 = torch.sum(x[:, 0, :] * x[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        o3 = torch.sum(x[:, 0, :] * nx[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o4 = torch.sum(x[:, 0, :] * nx[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        o5 = torch.sum(nx[:, 0, :] * x[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o6 = torch.sum(nx[:, 0, :] * x[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        o7 = torch.sum(nx[:, 0, :] * nx[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o8 = torch.sum(nx[:, 0, :] * nx[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        # return the stack of octants, should be (n, 8)
        return torch.stack([o1, o2, o3, o4, o5, o6, o7, o8], dim=1)
    ###
    #Testing/Validation Functions
    ###
    def p_bi(self, n, m, lam):
        if isinstance(n, float):
            n = np.array([n])
        if isinstance(m, float):
            m = np.array([m])
        _p_bi = binom(m, n) * np.power(lam, n) * np.power(1.0 - lam, m - n)
        _p_bi[np.logical_not(np.isfinite(_p_bi))] = 0.0
        return _p_bi

    def get_n1_n2(self,delta, m_1, m_2):
        r_1 = np.arange(0.0, m_1 + 0.5) / m_1
        r_2 = np.arange(0.0, m_2 + 0.5) / m_2
        X, Y = np.meshgrid(r_1, r_2)
        x = np.abs(X - Y)
        idx = np.argwhere(np.abs(x - delta) < 1E-6)
        n_1s = m_1 * r_1[idx[:, 1]]
        n_2s = m_2 * r_2[idx[:, 0]]
        return n_1s, n_2s

    def p_delta(self, delta, m_1, m_2, lam):
        _p_delta = 0.0
        n_1, n_2 = self.get_n1_n2(delta, m_1, m_2)
        _p_delta = np.sum(self.p_bi(n_1, m_1, lam) * self.p_bi(n_2, m_2, lam))
        return _p_delta
    
    def p_gtdelta(self, delta, m_1, m_2, lam):
        p_ltdelta = 0.0
        m = max([m_1, m_2])
        d_stars = np.arange(0.0, delta+1/m, 1/m)
        for d_star in d_stars:
            p_ltdelta += self.p_delta(d_star, m_1, m_2, lam)
        return 1.0 - p_ltdelta

    def m_line(self, delta, m_1, m_2):
        return max([m_1, m_2 * (1.0 - delta)])

    def p_D(self, pred=None, true=None):
        if pred is None:
            pred = self.pred
        if true is None:
            true = self.true
        m_1 = pred.shape[0]
        m_2 = true.shape[0]
        d = true.shape[1]
        D = self(pred, true).item()
        # round D to the nearest increment by the largest of the sample sizes
        m = m_1*m_2#max([m_1, m_2])
        D = np.round(m*D) / m
        #print(D, 'D')
        lambda_ik = self.M(true, torch.cat((pred, true))).numpy()
        # _p_D is the probability that every entry in M is less than or equal to
        # D
        _p_D = 1.0
        for i in range(lambda_ik.shape[0]):
            for k in range(lambda_ik.shape[1]):
                p_gtdelta = self.p_gtdelta(D, m_1, m_2, lambda_ik[i, k])
                _p_D *= 1.0 - p_gtdelta
        # we desire to know the probability that something will be larger than D
        return 1.0 - _p_D

    def p(self, pred=None, true=None):
        return self.p_D(pred=pred, true=true)

    def delta_pm(self, delta, m_1, m_2, n_1):
        delta_m = m_2 * ((n_1 / m_1) - delta)
        delta_p = m_2 * ((n_1 / m_1) + delta)
        return delta_p, delta_m

    def permute(self, pred=None, true=None, J=1_000):
        if pred is None:
            pred = self.pred
        if true is None:
            true = self.true
        all_pts = torch.cat((pred, true), dim=0)
        T = self(pred, true)
        T_ = torch.empty((J,))
        total_shape = pred.shape[0] + true.shape[0]
        for j in range(J):
            idx = torch.randperm(total_shape)
            idx1, idx2 = torch.chunk(idx, 2)
            _pred = all_pts[idx1]
            _true = all_pts[idx2]
            T_[j] = self(_pred, _true)
        return float(torch.sum(T < T_) + 1) / float(J + 2), T, T_
    ###
    # Utility Functions
    ###
    def softge(self, x, y):
        return (torch.tanh(10.0 * (x - y)) + 1.0) / 2.0

    def hardge(self, x, y):
        return torch.ge(x, y).long()
    
if __name__ == "__main__":
    p = torch.rand((100, 3))
    t = torch.rand((50, 3))

    calculation = ddKS()
    distance = calculation(p, t)
    print(f"The ddKS distance is {distance}")