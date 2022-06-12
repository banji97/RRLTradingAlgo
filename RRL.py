import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import numpy as np

class RRLModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.neuron = nn.Linear(m+1, 1, bias=True)
        nn.init.uniform_(self.neuron.weight, -.2, .2)
        nn.init.constant_(self.neuron.bias, 0)

    def forward(self, features):
        return torch.tanh(
            self.neuron(features)
        )

def Q_p(x):
    if x>5000:
        return .25*np.log(x)*.49088
    xs = np.array([0, 0.00001, 0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 
    0.0225, 0.0250, 0.0275, 0.0300, 0.0325, 0.0350, 0.0375, 0.0400, 0.0425, 0.0450, 0.0500, 0.0600, 0.0700, 0.0800, 
    0.0900, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 1.5000, 2.5000, 3.5000, 4.5000, 10.0000, 20.0000, 30.0000, 
    40.0000, 50.0000, 150.000, 250.000, 350.000, 450.000, 1000.00, 2000.00, 3000.00, 4000.00, 5000.00,])
    
    ys = np.array([0, 0.0028025, 0.0088623, 0.019690, 0.027694, 0.033789, 0.038896, 0.043372, 0.060721, 0.073808, 0.084693, 0.094171, 0.102651,
    0.110375, 0.117503, 0.124142, 0.130374, 0.136259, 0.141842, 0.147162, 0.152249, 0.157127, 0.161817, 0.166337,
    0.170702, 0.179015, 0.194248, 0.207999, 0.220581, 0.232212, 0.243050, 0.325071, 0.382016, 0.426452, 0.463159,
    0.668992, 0.775976, 0.849298, 0.905305, 1.088998, 1.253794, 1.351794, 1.421860, 1.476457, 1.747485, 1.874323,
    1.958037, 2.020630, 2.219765, 2.392826, 2.494109, 2.565985, 2.621743,])
    
    interp_func = interp1d(xs, ys)
    Q_n_predict = interp_func(x)
    return Q_n_predict

def Q_n(x):
    if x>5:
        return x+.5
    xs = np.array([0, 0.00001, 0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250, 0.0275, 
    0.0300, 0.0325, 0.0350, 0.0375, 0.0400, 0.0425, 0.0450, 0.0475, 0.0500, 0.0550, 0.0600, 0.0650, 0.0700, 0.0750, 0.0800, 
    0.0850, 0.0900, 0.0950, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 
    3.0000, 3.5000, 4.0000, 4.5000, 5.0000])
    
    ys = np.array([0, 0.0028025, 0.0088623, 0.019965, 0.028394, 0.034874, 0.040369, 0.045256, 0.064633, 0.079746, 0.092708, 0.104259, 0.114814,
    0.124608, 0.133772, 0.142429, 0.150739, 0.158565, 0.166229, 0.173756, 0.180793, 0.187739, 0.194489, 0.201094, 0.207572,
    0.213877, 0.220056, 0.231797, 0.243374, 0.254585, 0.265472, 0.276070, 0.286406, 0.296507, 0.306393, 0.316066, 0.325586,
    0.413136, 0.491599, 0.564333, 0.633007, 0.698849, 0.762455, 0.884593, 1.445520, 1.970740, 2.483960, 2.990940, 3.492520,
    3.995190, 4.492380, 4.990430, 5.498820,])

    interp_func = interp1d(xs, ys)
    Q_p_predict = interp_func(x)
    return Q_p_predict


def expected_max_drawdown(returns, sigma, time_horizon):
    """
    Determines the Expected Maximum drawdown
    Parameters
    ----------
    returns : Mean returns over the time period,T.
    sigma : Standard deviation of the mean returns.
    time_horizon : Time period.
    Returns
    -------
    E_MDD : Expected maximum drawdown, a risk based measure.
    """
    mean_copy = returns.clone().detach().numpy()
    sigma_copy = sigma.clone().detach().numpy()
    inner = (pow(mean_copy, 2) * time_horizon) / (2* pow(sigma_copy,2))
    #print(inner)#,returns,sigma,mean_copy,sigma_copy)
    if returns > 0:      
        E_MDD = torch.mul(torch.div((2 * torch.pow(sigma, 2)), returns), torch.tensor(Q_p(inner)))
        return E_MDD
    elif returns == 0:
        E_MDD = 1.2533 * sigma * np.sqrt(time_horizon)
        return E_MDD
    elif returns < 0:
        E_MDD = torch.mul(torch.div((-2 * torch.pow(sigma, 2)), returns), torch.tensor(Q_n(inner)))
        return E_MDD       
    
def calmar_ratio(returns: torch.Tensor, sigma, time_horizon):      
    """
    Determines the Calmar raito using expected maximum drawdown
    
    Parameters
    ----------
    returns : Returns over the time period,T.
    sigma : Standard deviation of the mean returns.
    time_horizon : Time period.
    Returns
    -------
    Calmar_ratio : Calmar ratio is a float.
    """
    
    mean_returns = torch.mean(returns, dim=-1)
    returns_sigma = torch.std(returns,dim=-1)
    calmar_ratio = mean_returns / expected_max_drawdown(returns = mean_returns, sigma = returns_sigma, time_horizon = time_horizon)
    return calmar_ratio

def sharpe_ratio(returns: torch.Tensor, eps: float = 1e-6):
    expected_return = torch.mean(returns, dim=-1)
    # The reference writeup used the biased STD estimator
    expected_squared_return = torch.mean(returns ** 2, dim=-1)
    sharpe = expected_return / (torch.sqrt(
        expected_squared_return - expected_return ** 2
    ) + eps)
    return sharpe


def reward_function(asset_returns: torch.Tensor, miu: float, delta: float, Ft: torch.Tensor, m: int, time_horizon):
    n = Ft.shape[-1] - 1
    returns = miu * (
        Ft[:n] * asset_returns[m:m+n]
    ) - (
        delta * torch.abs(Ft[1:] - Ft[:n])
    )
    returns_sigma = torch.std(returns)
    sharpe = sharpe_ratio(returns)
    calmar = calmar_ratio(returns, returns_sigma, time_horizon)
    # Returning Sharpe or Calmar ratio with change which function our algorithm is optimising for
    return returns, sharpe


def update_Ft(normalized_asset_returns: torch.Tensor, model: RRLModel):
    m = model.m
    t = normalized_asset_returns.shape[-1] - m
    Ft = torch.zeros(t + 1).to(normalized_asset_returns.device)
    for i in range(1, t):
        features = torch.cat([
            normalized_asset_returns[i-1:i+m-1], Ft[i-1:i]
        ])
        Ft[i] = model(features)
        # Can be adapted for long-only or short-only strategies
#          if Ft[i] > 0 :
#             Ft[i] = 1
#          elif Ft[i] < 0 :
#             Ft[i] = -1
    return Ft[1:]


def gradient_ascent(
        asset_returns: torch.Tensor,
        normalized_asset_returns: torch.Tensor,
        model: RRLModel,
        max_iter: int, lr: float, time_horizon):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    rewards = []
    
    for i in range(max_iter):
        optimizer.zero_grad()
        Ft = update_Ft(normalized_asset_returns, model)
        returns, reward = reward_function(asset_returns, miu=1., delta=0, Ft=Ft, m=model.m, time_horizon=time_horizon)
        (-1 * reward).backward()
        optimizer.step()
        rewards.append(reward.detach().cpu())
    return rewards, returns, Ft


def train(prices: torch.Tensor, m: int, t: int, delta: float = 0, max_iter: int = 100, lr: float = 0.1):
    assert len(prices.size()) == 1
    # asset returns are the ratio of the amount of change to the previous price
    asset_returns = (
        prices[1:] - prices[:-1]
    ).float() / prices[:-1]
    # to_be_predicted = prices.shape[0] - t - m
    scaler = StandardScaler()
    normalized_asset_returns = torch.tensor(scaler.fit_transform(
        asset_returns[:m+t][:, None].numpy()
    )[:, 0]).float()

    model = RRLModel(m)
    train_rewards, train_returns, train_Ft = gradient_ascent(
        asset_returns, normalized_asset_returns, model, max_iter, lr, time_horizon = t
    )

    normalized_asset_returns = torch.tensor(
        scaler.transform(asset_returns[t:][:, None].numpy())[:, 0]
    ).float()
    Ft_ahead = update_Ft(normalized_asset_returns, model)
    returns_ahead, reward_ahead = reward_function(asset_returns[t:], 1., delta, Ft_ahead, model.m, time_horizon= t)
    percentage_returns = (torch.exp(
        torch.log(1 + returns_ahead).cumsum(dim=-1)
    ) -1 ) * 100
    return {
        "valid_reward": reward_ahead,
        "valid_Ft": Ft_ahead,
        "valid_asset_returns": asset_returns[m+t:],
        "valid_asset_percentage_returns": (torch.exp(
            torch.log(1 + asset_returns[m+t:]).cumsum(dim=-1)
        ) - 1) * 100,
        "valid_percentage_returns": percentage_returns,
        "rewards_iter": train_rewards,
        "train_percentage_returns": (torch.exp(
            torch.log(1 + train_returns).cumsum(dim=-1)
        ) - 1) * 100,
        "train_Ft": train_Ft
    }