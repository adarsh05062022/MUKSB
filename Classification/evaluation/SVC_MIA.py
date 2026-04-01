"""SVM-based Membership Inference Attack (correctness, confidence, entropy, m-entropy, prob)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob         = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob     = 1 - p
    log_reverse_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    modified_probs   = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])
    prob, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = [t.to(next(model.parameters()).device) for t in batch]
                data, target = batch
            except Exception:
                device = (torch.device("cuda:0") if torch.cuda.is_available()
                          else torch.device("cpu"))
                data, target = batch[0].to(device), batch[1].to(device)
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
            targets.append(target)
    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_str = shadow_train.shape[0]; n_ste = shadow_test.shape[0]
    n_ttr = target_train.shape[0] if target_train is not None else 0
    n_tte = target_test.shape[0]  if target_test  is not None else 0

    X = torch.cat([shadow_train, shadow_test]).cpu().numpy().reshape(n_str + n_ste, -1)
    Y = np.concatenate([np.ones(n_str), np.zeros(n_ste)])
    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X, Y)
    accs = []
    if n_ttr > 0:
        accs.append(clf.predict(target_train.cpu().numpy().reshape(n_ttr, -1)).mean())
    if n_tte > 0:
        accs.append(1 - clf.predict(target_test.cpu().numpy().reshape(n_tte, -1)).mean())
    return np.mean(accs)


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    str_p, str_l = collect_prob(shadow_train, model)
    ste_p, ste_l = collect_prob(shadow_test,  model)
    ttr_p, ttr_l = collect_prob(target_train, model)
    tte_p, tte_l = collect_prob(target_test,  model)

    def _corr(p, l): return (torch.argmax(p, 1) == l).int()
    def _conf(p, l): return torch.gather(p, 1, l[:, None])

    str_corr = _corr(str_p, str_l); ste_corr = _corr(ste_p, ste_l)
    ttr_corr = _corr(ttr_p, ttr_l); tte_corr = _corr(tte_p, tte_l)
    str_conf = _conf(str_p, str_l); ste_conf = _conf(ste_p, ste_l)
    ttr_conf = _conf(ttr_p, ttr_l); tte_conf = _conf(tte_p, tte_l)
    str_entr = entropy(str_p); ste_entr = entropy(ste_p)
    ttr_entr = entropy(ttr_p); tte_entr = entropy(tte_p)
    str_me = m_entropy(str_p, str_l); ste_me = m_entropy(ste_p, ste_l)
    ttr_me = (m_entropy(ttr_p, ttr_l) if target_train is not None else ttr_entr)
    tte_me = (m_entropy(tte_p, tte_l) if target_test  is not None else tte_entr)

    m = {
        "correctness": SVC_fit_predict(str_corr, ste_corr, ttr_corr, tte_corr),
        "confidence":  SVC_fit_predict(str_conf, ste_conf, ttr_conf, tte_conf),
        "entropy":     SVC_fit_predict(str_entr, ste_entr, ttr_entr, tte_entr),
        "m_entropy":   SVC_fit_predict(str_me,   ste_me,   ttr_me,   tte_me),
        "prob":        SVC_fit_predict(str_p,    ste_p,    ttr_p,    tte_p),
    }
    print(m)
    return m


class MIA:
    """Alias kept for backward compatibility; use SVC_MIA directly."""
    def __init__(self, *a, **kw): pass
