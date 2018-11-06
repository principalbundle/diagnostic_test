import numpy as np, pandas as pd, random, re
from copy import deepcopy
from scipy.special import hyp2f1
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt
from graph import Graph
from node import FacNode, VarNode
from IPython import display

H = lambda x: sum(1. / np.arange(1, x + 1))

def bits(n):
    if n == 1:
        tmp = [[]]
    else:
        tmp = bits(n - 1)
    return [[1] + x for x in tmp] + [[0] + x for x in tmp]

def recur(margin_prob):
    if len(margin_prob) == 1:
        tmp = [1.]
    else:
        tmp = recur(margin_prob[1:])
    return [margin_prob[0] * x for x in tmp] + [(1 - margin_prob[0]) * x for x in tmp]

def recur(margin_prob):
    if len(margin_prob) == 1:
        tmp = [1.]
    else:
        tmp = recur(margin_prob[1:])
    return [margin_prob[0] * x for x in tmp] + [(1 - margin_prob[0]) * x for x in tmp]

def fpropagate(child, known_nodes, joint_prob):
    propagated, probT, probF = 0, deepcopy(joint_prob), deepcopy(joint_prob)

    parent_indices = [known_nodes.index(x) for x in child.parents]
    dummy_margin = np.ones(len(known_nodes)) / 2.
    seeds = bits(len(parent_indices))

    for i, s in enumerate(seeds):
        cond_true = child.cond_prob[i]
        dummy_margin[parent_indices] = s
        boolean = [x > 0 for x in recur(dummy_margin)]
        propagated += sum([x * y for x, y in zip(boolean, joint_prob)]) * cond_true
        for i, b in enumerate(boolean):
            if b:
                probT[i] *= cond_true
                probF[i] *= (1 - cond_true)

    child.prob = propagated
    return [child] + known_nodes, probT + probF

def entropy_gain(n, k):
    # H = lambda x: sum(1. / np.arange(1, x + 1))
    H1 = H(k) - np.log(k + 1)
    H2 = H(n - k) - np.log(n - k + 1)
    y = (k + 1) * H1 + (n - k + 1) * H2
    Hn = H(n + 1) - np.log(n + 2)
    return (y + 1) / (n + 2) - Hn

def generalised_entropy_gain(n, k, d, g):
    if abs(d - 1) < 1e-6: d = 1 + 1e-6
    a = hyp2f1(1, n - k + 1, n + 2, 1 - d)
    b = 0 if g == 0 else hyp2f1(1, n - k + 1, n + 2, 1 - d * g)
    x = (a * d - b * d * g) * ((1 - g) * d * a - (1 - g * d)) / (d - 1)  # * (k+1)/(n+2)#
    y = a * (1 - g) * d * (1 - a) / (d - 1)  # (n-k+1)/(n+2)#
    H1 = H(k) - np.log(k + 1)
    H2 = H(n - k) - np.log(n - k + 1)
    Hn = H(n + 1) - np.log(n + 2) - 1. / (n + 2)
    dS = (x * H1 + y * H2 - (x + y) * Hn)
    return dS  # , a*d - b*d*g, a

def demon(n, d, g, kgiven=0):
    ks = np.arange(n + 1)
    c1, c2 = [], []
    for k in ks:
        a = hyp2f1(1, n - k + 1, n + 2, 1 - d)
        b = 0 if g == 0 else hyp2f1(1, n - k + 1, n + 2, 1 - d * g)
        c1.append(a * d - b * d * g)
        c2.append(a)
    plt.plot(ks / n, c1, label='c1')
    plt.plot(ks / n, c2, label='c2')
    plt.xlabel('k / n')
    plt.ylabel('Effective n / n')
    plt.legend()

    plt.figure()
    S = [100 * generalised_entropy_gain(n, k, d, g) for k in ks]
    plt.plot(ks / n, S, label='S')
    plt.xlabel('k / n')
    plt.ylabel('information gain')

    plt.figure()
    dif = np.logspace(-1, 1, 100)
    S = [100 * generalised_entropy_gain(n, kgiven, d, g) for d in dif]
    plt.semilogx(dif, S, label='S')
    plt.xlabel('difficulty')
    plt.ylabel('information gain')

def iterscore(perf):
    n, k = 0, 0
    for key, val in perf.iterrows():
        nraw, kraw, d = 1., val.scaled_mark, val.d
        a = hyp2f1(1,n-k+1,n+2,1-d)
        ktmp = a*d*kraw
        k += ktmp
        n += (ktmp + a*(nraw-kraw))
    perf['n'] = n
    perf['k'] = k
    return perf.iloc[0, :]

class Subtopic:
    def __init__(self, name, prior=None, prior_k=0, prior_observation=0, parents=[], cond_prob=[]):
        self.name = name
        self.n = prior_observation
        self.k = prior_k
        self.parents = parents
        self.cond_prob = cond_prob
        self.children = []

        self.prob = prior
        self.n_prior = prior_observation
        self.k_infered = 0
        self.n_infered = 0

    def set_prior_probability(self, prior):
        self.prob = prior

    def set_cond_prob(self, if_all_parents_mastered):
        n = len(self.parents)
        if n > 0:
            self.cond_prob = [if_all_parents_mastered] + [0] * (2**n - 1)

    def set_children(self, children):
        self.children = children

    def reset_condprob(self, cond=[.8, .2]):
        self.cond_prob = cond[-1:] * 2 ** len(self.parents)
        self.cond_prob[0] = cond[0]

    def ppf(self, p, add=False):
        n = self.n
        k = self.k

        pnorm = norm.ppf(p)
        mean = (k + 1) / (n + 2)
        variance = mean * (k + 2) / (n + 3)
        skewness = variance * (k + 3) / (n + 4)

        variance -= mean ** 2
        skewness = (skewness - 3 * mean * variance - mean ** 3) / variance

        w = pnorm + skewness * (pnorm ** 2 - 1) / 6
        if add: w += skewness ** 2 * (5 * pnorm - 2 * pnorm ** 3) / 36
        percentile = mean + np.sqrt(variance) * w
        return -np.maximum(-1, -np.maximum(0, percentile))

    def intervals(self):
        n = self.n
        k = self.k
        print 'max-likelihood=%.2f, mean=%.2f, median=%.2f, 50-interval=[%.2f, %.2f], 80-interval=[%.2f, %.2f], 95-interval=[%.2f, %.2f]' % (
        k / n, (k + 1) / (n + 2), self.ppf(.5), self.ppf(.25), self.ppf(.75), self.ppf(.1), self.ppf(.9),
        self.ppf(.025), self.ppf(.975))

    def percent_among_peers(self, p=.5):
        l, r = self.ppf(p), self.ppf(1. - p)
        x = np.linspace(0, 1, 100)
        yprior = x ** (self.prob * self.n_prior) * (1 - x) ** ((1 - self.prob) * self.n_prior)
        yprior /= sum(yprior)
        return sum(yprior[x < l])

    def demon(self, p=.25, gaussian=True):
        plt.figure()
        l, r = self.ppf(p), self.ppf(1. - p)
        x = np.linspace(0, 1, 100)
        yprior = x ** (self.prob * self.n_prior) * (1 - x) ** ((1 - self.prob) * self.n_prior)
        yprior /= sum(yprior)

        if gaussian:
            xx = np.linspace(-3, 3, 100)
            L, R = sum(yprior[x < l]), sum(yprior[x > r])
            ll, rr = norm.ppf(L), -norm.ppf(R)
            gauss = np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi)
            plt.plot(xx, gauss)
            plt.fill_between(xx[(xx > ll) & (xx < rr)], gauss[(xx > ll) & (xx < rr)])
        else:
            y = x ** self.k * (1 - x) ** (self.n - self.k)
            y /= sum(y)
            plt.plot(x, y)
            plt.plot(x, yprior)
            plt.fill_between(x[(x > l) & (x < r)], yprior[(x > l) & (x < r)])

class Train_set:
    def __init__(self, hist_data, test_subset=-1, load_existing=None, combine=False):
        subs = pd.read_csv('basesubtopics.csv').rename(columns={'id': 'basesubtopic_id'}).set_index('basesubtopic_id')
        # subs_live = subs[subs['status'] <> 'error']

        subqs = pd.read_csv('basesubtopicrows.csv')
        num_qs = subqs.groupby('basesubtopic_id').apply(lambda x: float(x.order.iloc[-1]))
        subqs = subqs.set_index('basesubtopic_id')
        subqs['num_qs'] = num_qs
        subqs['d'] = np.exp(subqs['order'] / subqs['num_qs'] - .5)
        subqs = subqs.reset_index()

        subqs_live = subqs.join(subs, on='basesubtopic_id', how='left')
        subqs_live = subqs_live[subqs_live['status'] <> 'error']

        join = pd.read_csv('%s.csv' %(hist_data)).join(subqs_live.set_index('problemtemplate_id'), on='problemtemplate_id', how='left',
                         lsuffix='hist', rsuffix='sub')

        edges = pd.read_csv('requirements.csv')
        hist = join.iloc[:test_subset, :].groupby('basesubtopic_id').apply(iterscore).groupby('user_id')

        nodes = {}
        # subqs_live.set_index('basesubtopic_id', inplace=True)
        for key in subqs_live.basesubtopic_id.unique():
            nodes[key] = subtopic(key, parents=[])

        for key, val in edges.iterrows():
            try:
                nodes[val.leads_to_basesubtopic_id].parents.append(nodes[val.basesubtopic_id])
            except KeyError:
                pass

        for key, val in edges.iterrows():
            try:
                nodes[val.basesubtopic_id].children.append(nodes[val.leads_to_basesubtopic_id])
            except KeyError:
                pass

        self.subtopics = len(nodes)
        self.students = len(hist.user_id.unique())

        n = {name: np.zeros(self.students) for name, node in nodes.iteritems()}
        k = {name: np.zeros(self.students) for name, node in nodes.iteritems()}
        for i, (student, group) in enumerate(hist):
            for key, row in group.iterrows():
                n[key][i] = row.n
                k[key][i] = row.k

        self.tree = network(nodes.values(), num_items=1)
        self.tree.set_items(subqs)

        if combine:
            self.nodes = {node: np.array([.5, sum(n[name]), sum(k[name])])[:, None] for name, node in nodes.iteritems()}
            self.students = 1
        else:
            self.nodes = {node: np.vstack((.5 * np.ones(self.students), n[name], k[name])) for name, node in nodes.iteritems()}

        if load_existing is None:
            self.tree.init_probs()
            self.tree.init_varfac()
        else:
            self.load_train(load_existing)
        self.performance = 0

    def reset(self):
        self.performance = 0
        self.tree.init_probs()
        self.tree.init_varfac()
        for node in self.nodes.iterkeys():
            self.nodes[node][0, :] = .5

    def calc_prior_n(self, damping_factor=5.):
        n_infer = np.zeros((self.students, self.subtopics))
        k_infer = np.zeros((self.students, self.subtopics))
        for i, node in enumerate(tree.nodes):
            n = self.nodes[node][1, :]
            p = self.nodes[node][2, :] / (n + 1e-12)
            for j in range(self.subtopics):
                nmeasure = n * self.tree.damp_factor[i, j]
                pmeasure = p * self.tree.pair_cond_true[i, j] + (1 - p) * self.tree.pair_cond_false[i, j]
                n_infer[:, j] += nmeasure
                k_infer[:, j] += nmeasure * pmeasure

        alpha, beta = k_infer + 1., n_infer - k_infer + 1.
        means = alpha / (alpha + beta)
        weights = (n_infer + 1.)
        EX2 = np.sum(weights * (alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1) + means ** 2), axis=0) / np.sum(
            weights, axis=0)
        EX = np.sum(weights * means, axis=0) / np.sum(weights, axis=0)
        variance = EX2 - EX ** 2

        p = np.array([node.prob for node in self.tree.nodes])
        b = 7. - p * (1. - p) / variance
        c = 16. - 1. / variance
        d = 12. - 1. / variance
        D0 = b ** 2 - 3. * c
        D1 = 2. * b ** 3 - 9. * b * c + 27. * d
        C = np.cbrt((D1 + np.sqrt(D1 ** 2 - 4. * D0 ** 3)) / 2.)
        ns = -(b + C + D0 / C) / 3.

        ns /= damping_factor
        for i, node in enumerate(tree.nodes):
            node.n_prior = 0 if np.isnan(ns[i]) else ns[i]

    def train_network(self, maxiter=10, student_batch=10, observed_only=False, plot=True, autosave=None, damping_factor=5.):
        student_batch = min(student_batch, self.students)
        for i in xrange(maxiter):

            students = random.sample(range(self.students), student_batch)
            for j in students:
                print 'student %d' %(j)
                for node, array in self.nodes.iteritems():
                    node.prob, node.n, node.k = array[:, j]

                for node in self.tree.heads:
                    self.tree.varfacnodes[node][1].P = np.array([node.prob, 1-node.prob])

                self.performance += self.tree.gradient_descent(plot=False, eps=.1, observed_only=observed_only, maxiter=5, head_batch=50,
                                 deriv_batch=20, head_rate=.01, deriv_rate=.001)

                for node in self.tree.heads:
                    self.nodes[node][0, j] = node.prob

            # self.calc_prior_n(damping_factor)
            if autosave is not None:
                self.save_train(autosave)
            if plot:
                self.train_plot(self.performance)

    def save_train(self, autosave):
        self.tree.save_model(autosave + '_cond')
        prob = {node.name: array[0, :] for node, array in self.nodes.iteritems()}
        pd.DataFrame(prob).to_csv('%s.csv' %(autosave + '_head'))

    def load_train(self, autosave):
        self.tree.load_model(autosave + '_cond')
        df = pd.read_csv(autosave + '_head.csv')
        for node, array in self.nodes.iteritems():
            array[0, :] = df[str(node.name)].as_matrix()

    def train_plot(self, loglikelihood):
        if not hasattr(self, 'loglikelihoods'):
            self.loglikelihoods = []
        self.loglikelihoods += [loglikelihood]
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10, 6))
        plt.plot(self.loglikelihoods, label='log-likelihood')
        plt.legend()

class Item:
    def __init__(self, name, num_of_question=1.):
        self.name = name
        self.num_of_question = num_of_question
        self.property = []

    def link_subtopic(self, node, difficulty=1., types=.0, costs=1.):
        self.property.append((node, difficulty, types, costs))

    def correct(self):
        self.score = 1.

    def wrong(self):
        self.score = 0.

class Network:
    def __init__(self, node_list=[], num_items=1, item_list=None, decay_factor=0, axlr=1):
        self.axlr = axlr
        self.decay_factor = decay_factor
        self.num_nodes = len(node_list)
        self.nodes = node_list

        if item_list is None:
            self.num_items = num_items
            ids = np.repeat(np.arange(self.num_items)[None,:], self.num_nodes, axis=0)
            difficulty = np.exp(np.random.randn(self.num_nodes, self.num_items))
            types = np.random.randint(low=0, high=3, size=(self.num_nodes, self.num_items)) / 4.
            costs = np.exp(np.random.randn(self.num_nodes, self.num_items) * .1)
            # self.set_children()
            self.items = {self.nodes[i]: np.vstack((ids[i, :], difficulty[i, :], types[i, :], costs[i, :])) for i in
                          xrange(self.num_nodes)}
        else:
            self.items = item_list

    def find_node(self, node_name):
        for node in self.nodes:
            if node.name == node_name:
                return node
        raise Exception('node not found!')

    def set_items(self, subqs):
        if type(subqs)==str:
            subqs = pd.read_csv('%s.csv' %(subqs))
            num_qs = subqs.groupby('basesubtopic_id').apply(lambda x: float(x.order.iloc[-1]))
            subqs = subqs.set_index('basesubtopic_id')
            subqs['num_qs'] = num_qs
            subqs['d'] = np.exp(subqs['order'] / subqs['num_qs'] - .5)
            subqs = subqs.reset_index()

        for key, val in self.items.iteritems():
            shortanswer = subqs.set_index('basesubtopic_id').ix[key.name][['problemtemplate_id', 'd']].as_matrix().T
            multichoice = np.vstack((shortanswer, np.ones(shortanswer.shape) * np.array([[.25], [.75]])))
            self.items[key] = np.hstack((np.vstack((shortanswer, np.ones(shortanswer.shape) * np.array([[0], [1]]))), multichoice))

    def save_model(self, model_name):
        prob = [node.prob for node in self.nodes]
        n_prior = [node.n_prior for node in self.nodes]
        cond_prob = [node.cond_prob for node in self.nodes]
        names = [node.name for node in self.nodes]
        df = pd.DataFrame({'name': names, 'prob': prob, 'cond_prob': cond_prob, 'n_prior': n_prior}).set_index('name')
        df.to_csv('%s.csv' %(model_name))

    def load_model(self, model_name):
        reg = lambda x: np.array(map(float, re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", x)))
        df = pd.read_csv('%s.csv' %(model_name)).set_index('name')
        for name, val in df.iterrows():
            found = False
            for node in self.nodes:
                if node.name == name:
                    found = True
                    break
            if not found:
                print '%s not found!' % (name)
                continue
            node.prob = val.prob
            node.n_prior = val.n_prior
            node.k = val.prob * val.n_prior
            node.n = val.n_prior
            node.cond_prob = reg(val.cond_prob)
            if len(node.cond_prob) > 2**len(node.parents):
                node.reset_condprob()
        self.init_varfac()

    def load_pair(self, pair_name):
        df = pd.read_csv('%s_true.csv' %(pair_name))
        self.pair_cond_true = df.as_matrix()[:, 1:]
        df = pd.read_csv('%s_false.csv' %(pair_name))
        self.pair_cond_false = df.as_matrix()[:, 1:]

    def init_probs(self):
        for node in self.nodes:
            node.prob = .5
            if len(node.parents) > 0:
                node.reset_condprob()

    def init_varfac(self):
        self.heads, self.derivs = self.head_nodes()
        self.G = Graph()
        self.varfacnodes = {node: [self.G.addVarNode(node.name, 2)] for node in self.nodes}
        # head_names = [somenode.name for somenode in known_nodes]
        for node, varfac in self.varfacnodes.iteritems():
            if node in self.heads:
                p = node.prob
                fac = self.G.addFacNode(np.array([p, 1-p]), varfac[0])
            else:
                subset = [self.varfacnodes[somenode][0] for somenode in node.parents]
                # print subset, node.cond_prob
                condp = np.array(node.cond_prob).reshape([2] * len(subset) + [1])
                condp = np.concatenate((condp, 1-condp), axis=-1)
                fac = self.G.addFacNode(condp, *(subset + [varfac[0]]))
            if len(varfac) < 2:
                varfac.append(fac)

    def next_item(self, subtopic):
        ids, difficulty, types, costs = self.items[subtopic]
        n, k = subtopic.n, subtopic.k
        dS = [generalised_entropy_gain(n, k, d, g) for d, g in zip(difficulty, types)]
        dS /= costs
        maxind = np.argmax(dS)

        mask = list(xrange(maxind)) + list(xrange(maxind + 1, len(difficulty)))
        # print maxind, mask
        self.items[subtopic] = np.vstack((ids[mask], difficulty[mask], types[mask], costs[mask]))  # remove after test

        d, g = difficulty[maxind], types[maxind]
        a = hyp2f1(1, n - k + 1, n + 2, 1 - d)
        b = 0 if g == 0 else hyp2f1(1, n - k + 1, n + 2, 1 - d * g)
        return (ids[maxind], d, g, costs[maxind]), a * d - b * d * g, a

    def head_nodes(self):
        heads = []
        derivs = []
        for node in self.nodes:
            if len(node.parents) == 0:
                heads.append(node)
            else:
                derivs.append(node)
        return heads, derivs

    def set_parents(self):
        for node in self.nodes:
            node.parents = []
            for candidate in self.nodes:
                if node in candidate.children:
                    node.parents.append(candidate)

    def set_children(self):
        for node in self.nodes:
            node.children = []
            for candidate in self.nodes:
                if node in candidate.parents:
                    node.children.append(candidate)

    def forward_propagate(self, known_nodes):
        joint_prob = recur([x.prob for x in known_nodes])
        for i in xrange(self.num_nodes - len(known_nodes)):
            for known_node in known_nodes:
                for child in known_node.children:
                    if (child.prob is None) and all(x in known_nodes for x in child.parents):
                        known_nodes, joint_prob = fpropagate(child, known_nodes, joint_prob)
        self.joint_prob = joint_prob
        self.nodes = known_nodes

    def sumproduct_propagate(self, known_nodes=[], reset=False):
        reset_dict = {}
        for node in known_nodes:
            if reset:
                reset_dict[node] = self.varfacnodes[node][1].P
            if node in self.heads:
                p = node.prob
                self.varfacnodes[node][1].P = np.array([p, 1-p])
            else:
                # subset = [self.varfacnodes[somenode][0] for somenode in node.parents]
                condp = np.array(node.cond_prob).reshape([2] * len(node.parents) + [1])
                self.varfacnodes[node][1].P = np.concatenate((condp, 1-condp), axis=-1)

        self.G.converged = False
        marg = self.G.marginals()
        for node in self.nodes:
            node.prob = marg[node.name][0][0]

        if reset:
            for node in known_nodes:
                self.varfacnodes[node][1].P = reset_dict[node]

    def forward_pair(self):
        self.pair_cond_true, self.pair_cond_false = np.ones((self.num_nodes, self.num_nodes)), np.zeros(
            (self.num_nodes, self.num_nodes))
        dummy_margin = np.ones(self.num_nodes) / 2.

        for tf, mat in zip([1, 0], [self.pair_cond_true, self.pair_cond_false]):
            for i in xrange(self.num_nodes):
                dummy_margin[i] = tf
                boolean = [x > 0 for x in recur(dummy_margin)]
                margin = sum([x * y for x, y in zip(boolean, self.joint_prob)])
                for j in xrange(self.num_nodes):
                    if i == j: continue
                    dummy_margin[j] = 1
                    boolean = [x > 0 for x in recur(dummy_margin)]
                    mat[i, j] = sum([x * y for x, y in zip(boolean, self.joint_prob)]) / margin
                    dummy_margin[j] = .5
                dummy_margin[i] = .5

    def sumproduct_pair(self, save=None):
        self.pair_cond_true, self.pair_cond_false = np.ones((self.num_nodes, self.num_nodes)), np.zeros(
            (self.num_nodes, self.num_nodes))

        for p in [1, 0]:
            for i, ni in enumerate(self.nodes[::-1]):
                self.G.converged = False
                if ni in self.heads:
                    old_p = self.varfacnodes[ni][1].P
                    self.varfacnodes[ni][1].P = np.array([float(p), 1-float(p)])
                else:
                    self.G.var[ni.name].condition(1-p)
                    self.G.marginals(1)
                    # self.varfacnodes[ni][0].dim = 1
                    # condp = np.array(ni.cond_prob).reshape([2] * len(ni.parents) + [1])
                    # if p > 0:
                    #     self.varfacnodes[ni][1].P = condp
                    # else:
                    #     self.varfacnodes[ni][1].P = 1. - condp
                    self.G.converged = False

                marg = self.G.marginals()
                for j, nj in enumerate(self.nodes[::-1]):
                    if p > 0:
                        self.pair_cond_true[i, j] = 1. if i==j else marg[nj.name][0][0]
                    else:
                        # print marg[nj.name][0][0], i, j
                        self.pair_cond_false[i, j] = 0. if i==j else marg[nj.name][0][0]

                if ni in self.heads:
                    self.varfacnodes[ni][1].P = old_p
                else:
                    self.G.var[ni.name].reset()
                    # self.varfacnodes[ni][0].dim = 2
                    # self.varfacnodes[ni][1].P = np.concatenate((condp, 1-condp), axis=-1)
                print ni.name,# marg[nj.name][0][0],
            if save is not None:
                if p > 0:
                    pd.DataFrame(self.pair_cond_true).to_csv('%s_true.csv' %(save))
                else:
                    pd.DataFrame(self.pair_cond_false).to_csv('%s_false.csv' %(save))

    def observation_damp_factor(self):
        self.damp_factor = np.ones((self.num_nodes, self.num_nodes))
        for i, from_node in enumerate(self.nodes):
            for j, to_node in enumerate(self.nodes):
                if i == j: continue
                f0, f1, p = self.pair_cond_false[i, j], self.pair_cond_true[i, j], from_node.prob
                expect = p * f1 + (1 - p) * f0
                tmp = p * f1 * (1 - f1) + (1 - p) * f0 * (1 - f0)
                self.damp_factor[i, j] = max(0, 1 - tmp / expect / (1 - expect))
        self.damp_factor = self.damp_factor ** self.decay_factor

    def propagate_observation(self, node, k, n=1, silent=False):
        if not silent: print '%s: n=%d, k=%d' % (node.name, n, k)
        i = self.nodes.index(node)
        p = k / float(n)
        for j, node in enumerate(self.nodes):
            nmeasure = n * self.damp_factor[i, j]
            pmeasure = p * self.pair_cond_true[i, j] + (1 - p) * self.pair_cond_false[i, j]
            node.n += nmeasure
            if k == n:
                node.k += max(pmeasure * nmeasure, node.k * nmeasure / (node.n - nmeasure))
            elif k == 0:
                node.k += min(pmeasure * nmeasure, node.k * nmeasure / (node.n - nmeasure))
            else:
                node.k += pmeasure * nmeasure

    def propagate_observation_item(self, item, k, n=1., silent=False):
        def coeffs(n, k, d, g):
            a = hyp2f1(1, n - k + 1, n + 2, 1 - d)
            b = 0 if g == 0 else hyp2f1(1, n - k + 1, n + 2, 1 - d * g)
            return a * d - b * d * g, a

        deltak, deltan = np.zeros(self.num_nodes), np.zeros(self.num_nodes)
        for node, difficulty, types, costs in item.property:
            c1, c2 = coeffs(n, k, difficulty, types)
            effk = c1 * k
            effn = c1 * k + c2 * (n - k)

            if not silent: print '%s: n=%.4f, k=%.4f' % (node.name, effn, effk)
            i = self.nodes.index(node)
            p = effk / float(effn)
            for j, node in enumerate(self.nodes):
                nmeasure = effn * self.damp_factor[i, j]
                pmeasure = p * self.pair_cond_true[i, j] + (1 - p) * self.pair_cond_false[i, j]
                # deltan[j] += nmeasure * self.axlr
                # deltak[j] += pmeasure * nmeasure * self.axlr
                if k == n:
                    deltan[j] += nmeasure * self.axlr * (pmeasure >= 1)
                    deltak[j] += pmeasure * nmeasure * self.axlr * (pmeasure >= 1)
                elif k == 0:
                    deltan[j] += nmeasure * self.axlr * (pmeasure <= 0)
                    deltak[j] += pmeasure * nmeasure * self.axlr * (pmeasure <= 0)
                else:
                    deltan[j] += nmeasure * self.axlr
                    deltak[j] += pmeasure * nmeasure * self.axlr

        for j, node in enumerate(self.nodes):
            if deltan[j] <= 0:
                continue
            deltakn = deltak[j] / deltan[j]
            # print node.name, deltak[j], deltan[j], deltakn
            if k == n:
                node.k += max(deltak[j], node.k * deltan[j] / node.n)
                node.n += deltan[j]
                kn = node.k / node.n if node.n > 0 else 0
                # node.k += max(deltakn, kn) * deltak[j]
                # node.n += deltak[j]
            elif k == 0:
                node.k += min(deltak[j], node.k * deltan[j] / node.n)
                node.n += deltan[j]
                kn = node.k / node.n if node.n > 0 else 1
                # node.k += min(deltakn, kn) * (deltan[j] - deltak[j])
                # node.n += deltan[j] - deltak[j]
            else:
                node.k += deltak[j]
                node.n += deltan[j]

    def reset_nk(self, n_prior=0):
        for node in self.nodes:
            node.n_prior = n_prior
            node.n = node.n_prior
            node.k = node.n * node.prob

    def final_score(self):
        self.reset_nk()
        for item in self.items:
            n = item.num_of_question
            k = item.num_of_question * item.score
            self.propagate_observation_item(item, k, n, silent=True)

    def single_update(self, item, silent=True):
        n = item.num_of_question
        k = item.num_of_question * item.score
        self.propagate_observation_item(item, k, n, silent=silent)

    def gradient_head(self, perturb_node):
        p = perturb_node.prob
        perturb_node.prob = 0
        self.sumproduct_propagate(known_nodes=[perturb_node], reset=True)
        perturb_prob = np.array([node.prob for node in self.nodes])
        gradient = sum((self.base_prob - perturb_prob) * self.base_gradient) / p
        perturb_node.prob = p
        return gradient

    def gradient_cond_prob(self, perturb_node):
        gradient = np.zeros(len(perturb_node.cond_prob))
        for i, cp in enumerate(perturb_node.cond_prob):
            perturb_node.cond_prob[i] = 0
            self.sumproduct_propagate(known_nodes=[perturb_node], reset=True)
            perturb_prob = np.array([node.prob for node in self.nodes])
            gradient[i] = sum((self.base_prob - perturb_prob) * self.base_gradient) / cp
            perturb_node.cond_prob[i] = cp
        return gradient

    def train_plot(self, loglikelihood):
        if not hasattr(self, 'loglikelihoods'):
            self.loglikelihoods = []
        self.loglikelihoods += [loglikelihood]
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10, 6))
        plt.plot(self.loglikelihoods, label='log-likelihood')
        plt.legend()

    def gradient_descent(self, debug=True, plot=True, eps_base=.01, eps=.01, observed_only=False, maxiter=20, head_batch=5, deriv_batch=10, head_rate=.01, deriv_rate=.01, accel=1e4):
        base_n, base_k = np.array([node.n for node in self.nodes]), np.array([node.n for node in self.nodes])

        if observed_only:
            observed_heads, observed_derive = [] , []
            for node in self.nodes:
                if node.n > 0:
                    if node in self.heads:
                        observed_heads.append(node)
                    else:
                        observed_derive.append(node)
        else:
            observed_heads = self.heads
            observed_derive = self.derivs

        head_all = len(observed_heads)
        deriv_all = len(observed_derive)
        head_batch = min(head_batch, head_all)
        deriv_batch = min(deriv_batch, deriv_all)

        self.base_prob = np.array([node.prob for node in self.nodes])
        start_performance = sum(base_k * np.log(self.base_prob) + (base_n - base_k) * np.log(1. - self.base_prob))
        self.sumproduct_propagate()
        self.base_prob = np.array([node.prob for node in self.nodes])

        head_prob = np.zeros(head_batch)
        for i in xrange(maxiter):
            self.base_gradient = base_k / self.base_prob - (base_n - base_k) / (1. - self.base_prob)

            heads = random.sample(range(head_all), head_batch)
            for count, j in enumerate(heads):
                node = observed_heads[j]
                # print node.n, node.k, head_rate
                if node.n > 0:
                    prob = node.prob + head_rate * self.gradient_head(node)
                else:
                    prob = node.prob + accel * head_rate * self.gradient_head(node)
                if prob < eps_base:
                    prob = eps_base
                elif prob > 1.-eps_base:
                    prob = 1.-eps_base
                if debug: print '%s: %.2f --> %.2f' % (node.name, node.prob, prob)
                head_prob[count] = prob

            derivs = random.sample(range(deriv_all), deriv_batch)
            for j in derivs:
                node = observed_derive[j]
                if debug: print '%s: %r --> ' %(node.name, node.cond_prob),
                # print node.n, deriv_rate
                if node.n > 0:
                    node.cond_prob += deriv_rate * self.gradient_cond_prob(node)
                else:
                    node.cond_prob += accel * deriv_rate * self.gradient_cond_prob(node)
                if any(node.cond_prob < eps):
                    node.cond_prob = np.maximum(eps, node.cond_prob)
                elif any(node.cond_prob > 1.-eps):
                    node.cond_prob = -np.maximum(-1.+eps, -node.cond_prob)
                if debug: print '%r' %(node.cond_prob)

            for count, j in enumerate(heads):
                node = observed_heads[j]
                node.prob = head_prob[count]
                self.varfacnodes[node][1].P = np.array([node.prob, 1 - node.prob])

            for j in derivs:
                node = observed_derive[j]
                condp = np.array(node.cond_prob).reshape([2] * len(node.parents) + [1])
                self.varfacnodes[node][1].P = np.concatenate((condp, 1-condp), axis=-1)

            self.sumproduct_propagate()
            self.base_prob = np.array([node.prob for node in self.nodes])
            if plot:
                self.train_plot(sum(base_k * np.log(self.base_prob) + (base_n - base_k) * np.log(1. - self.base_prob)))

        end_performance = sum(base_k * np.log(self.base_prob) + (base_n - base_k) * np.log(1. - self.base_prob))
        return end_performance - start_performance

    def print_network(self):
        for node in self.nodes[::-1]:
            print '%s: prob=%.2f, n=%.1f' % (node.name, node.prob, node.n)
        print '---------------------'

    def draw_network(self, m=2, pos=None):
        edgelist = []
        probdict = {}
        nodesizes = {}
        position = {}
        observations = {}
        for i, node in enumerate(self.nodes[::-1]):
            l, u = node.ppf(.1), node.ppf(.9)
            c, w = .5 * (l + u), .5 * (u - l)
            edgelist += [(node.name, x.name) for x in node.children]
            probdict[node.name] = c
            nodesizes[node.name] = 100 + 5000 * w
            position[node.name] = (i / m, i % m) if pos is None else pos[node.name]
            observations[node.name] = '%.2f+/-%.2f' % (c, w)

        G = nx.DiGraph()
        G.add_edges_from(edgelist)
        values = [probdict.get(node) for node in G.nodes()]
        sizes = [nodesizes.get(node) for node in G.nodes()]

        plt.figure(figsize=(16,8))
        nx.draw_networkx_nodes(G, position, node_size=sizes, cmap=plt.get_cmap('Blues'), node_color=values, vmin=-0.1, vmax=1)
        nx.draw_networkx_labels(G, position, {x: x for x in G.nodes()}, font_size=12, font_color='r')
        nx.draw_networkx_labels(G, {key: (x[0] , x[1]+.3) for key, x in position.iteritems()}, observations,
                                font_size=12, font_color='k')
        nx.draw_networkx_edges(G, position, edgelist=edgelist, edge_color='r', arrows=True, alpha=0.5)
        # plt.xlim((-.15,.9))
        plt.show()

    def next_subtopic(self):
        increments, total_gains = [], []
        for node in self.nodes:
            increments.append(entropy_gain(node.n, node.k))
        for i, node in enumerate(self.nodes):
            total_gains.append(sum([x * y for x, y in zip(increments, self.damp_factor[i, :])]))
        optimal = total_gains.index(max(total_gains))
        return self.nodes[optimal]

    def recommend(self):
        # increments, total_gains = [], []
        k_pre = np.array([node.prob * node.n_prior + node.k_infered for node in self.nodes])
        n_pre = np.array([node.n_prior + node.n_infered for node in self.nodes])
        prob = np.array([node.prob for node in self.nodes])
        # consistency = sum(k_pre * np.log(prob) + (n_pre - k_pre) * np.log(1 - prob))
        grad = k_pre / prob - (n_pre - k_pre) / (1 - prob)
        htol = sorted(range(len(grad)), key=lambda k: -grad[k])
        return self.nodes[0], [self.nodes[i].name for i in htol]
