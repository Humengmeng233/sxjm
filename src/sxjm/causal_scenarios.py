"""Past-only forecast combination, split calibration, and paired trajectory blocks."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ScenarioBundle:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    net: np.ndarray
    prices: np.ndarray
    probability: np.ndarray
    source_indices: np.ndarray
    calibration_count: int


def _causal_centers(actual,same_weekday_count=8):
    """Combine recent and same-weekday forecasts using PREVIOUS forecast errors."""
    n, horizon = actual.shape
    components = np.zeros((n,2,horizon))
    prediction = np.zeros_like(actual)
    for day in range(1,n):
        recent = actual[max(0,day-7):day]
        weekday = actual[np.arange(day-7,max(-1,day-7*same_weekday_count-1),-7)] if day >= 7 else recent
        components[day,0] = np.median(recent,axis=0)
        components[day,1] = np.median(weekday,axis=0)
        begin = max(1,day-14)
        if day > begin:
            error = np.mean(np.abs(components[begin:day]-actual[begin:day,None,:]),axis=(0,2))
            weights = 1/np.maximum(error,1.)
            weights /= weights.sum()
        else:
            weights = np.array([.5,.5])
        prediction[day] = weights@components[day]
    return prediction


class CausalScenarioFactory:
    def __init__(self, data, config, scenario_count=8):
        if scenario_count < 2:
            raise ValueError("At least two support scenarios required")
        self.data, self.config, self.count = data, config, scenario_count
        # Precomputation is a chronological recurrence; each prediction uses < day.
        self.net_center = _causal_centers(data.net_kwh,config.same_weekday_count)
        self.load_center = _causal_centers(data.load_kwh,config.same_weekday_count)
        self.price_center = _causal_centers(data.variable_price,config.same_weekday_count)
        self.intraday = np.zeros_like(data.pv_forecast_kwh)
        for day in range(1,len(data.dates)):
            for release, slot in enumerate((0,36,72,108)):
                ratio = 1.
                if slot:
                    ratio = np.clip(np.median(data.load_kwh[day,:slot]/np.maximum(self.load_center[day,:slot],1.)),.8,1.2)
                self.intraday[day,release,slot:] = ratio*self.load_center[day,slot:]-data.pv_forecast_kwh[day,release,slot:]

    def get(self, day, release=0, *, pv_information=False, variable_price=False, unknown_price=False):
        slot = (0,36,72,108)[release]
        if day < 14:
            raise ValueError("Need at least fourteen days for forecast calibration")
        indices = np.arange(max(7,day-self.config.recent_window_days),day)
        centers = self.intraday[:,release] if pv_information else self.net_center
        center = centers[day,slot:].copy()
        residual = self.data.net_kwh[indices,slot:]-centers[indices,slot:]
        # Fit and calibrate on disjoint historical days; never use target-day errors.
        cut = max(2,len(indices)//2)
        fit, calibration = residual[:cut], residual[cut:]
        tail = (1-self.config.conformal_alpha)/2
        low, high = np.quantile(fit,[tail,1-tail],axis=0)
        scores = np.maximum(low-calibration,calibration-high)
        rank = min(len(calibration),int(np.ceil((len(calibration)+1)*self.config.conformal_alpha)))
        correction = np.sort(scores,axis=0)[rank-1]
        # Clipping negative correction avoids an inverted/overly narrow band.
        correction = np.maximum(correction,0.)
        lower, upper = center+low-correction, center+high+correction
        paths = center+residual
        if variable_price and unknown_price:
            price_residual = self.data.variable_price[indices,slot:]-self.price_center[indices,slot:]
            price_paths = np.maximum(.01,self.price_center[day,slot:]+price_residual)
        else:
            known = self.data.variable_price[day,slot:] if variable_price else self.data.fixed_price[slot:]
            price_paths = np.broadcast_to(known,paths.shape).copy()
        # Empirical copula = jointly resample SAME daily net/price residual row.
        # Deterministic medoid support reduction; probability = cluster frequency.
        scale_n = max(float(np.std(paths,axis=0).mean()),1.)
        scale_p = max(float(np.std(price_paths,axis=0).mean()),.01)
        distance = np.abs(paths[:,None]-paths[None,:]).mean(axis=2)/scale_n
        distance += np.abs(price_paths[:,None]-price_paths[None,:]).mean(axis=2)/scale_p
        chosen = [int(np.argmin(distance.sum(axis=1)))]
        while len(chosen) < min(self.count,len(indices)):
            nearest = distance[:,chosen].min(axis=1); nearest[chosen] = -1.
            chosen.append(int(np.argmax(nearest)))
        assignment = np.argmin(distance[:,chosen],axis=1)
        probability = np.bincount(assignment,minlength=len(chosen))/len(indices)
        keep = probability > 0
        chosen = np.asarray(chosen)[keep]; probability = probability[keep]
        return ScenarioBundle(center,lower,upper,paths[chosen],price_paths[chosen],probability,indices[chosen],len(calibration))
