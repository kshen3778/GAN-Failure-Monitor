from scipy.optimize import curve_fit
import numpy as np

class GANMonitor:
    def __init__(self, total_iterations, warmup_time, failure_duration=0.15, bottom_threshold=0.05, top_threshold=0.95, curve_variance=0.05):
        self.bottom_threshold = bottom_threshold
        self.top_threshold = top_threshold
        self.total_iterations = total_iterations
        self.warmup_time = warmup_time
        self.data = {
            "disc_loss_1": [],
            "disc_loss_2": [],
            "gen_loss": [],
            "acc_1": [],
            "acc_2": []
        }

    def set_value(self, name, values):
      data[name] = values

    def log_value(self, name, value):
      data[name].append(value)

    def exponential(x, a, b, c):
      return a*np.exp(b*x) + c

    def logarithm(x, a, b, c):
      return a*np.log(b*x) + c
    
    def fit_exp_curve(values):
      x_values = range(1, len(values) + 1)
      pars, cov = curve_fit(f=exponential, xdata=x_values, ydata=values, p0=[0, 0, 0], bounds=(-np.inf, np.inf))
      stdevs = np.sqrt(np.diag(cov_loss))
      return pars, stdevs
    
    def fit_exp_curve(values):
      x_values = range(1, len(values) + 1)
      pars, cov = curve_fit(f=logarithm, xdata=x_values, ydata=values, bounds=([-np.inf, 0.00001, -np.inf], [np.inf, np.inf, np.inf]))
      stdevs = np.sqrt(np.diag(cov_loss))
      return pars, stdevs

    #check to see if loss falls to near zero
    def checkDecreasingLoss(d_loss):
      valueTreshold = bottom_threshold
      stepThreshold = int(self.warmup_time * self.total_iterations)
      durationThreshold = int(self.failure_duration * self.total_iterations)
      count = 0
      for i in range(len(d_loss)):
        if(i > stepThreshold and d_loss[i] < valueThreshold):
          count += 1
          if(count > durationThreshold): #possible failure case detected
            return True 
        else:
          count = 0
      
      return False

    #check if fits increasing exponentials curve
    def checkExpCurveIncrease(values):

      params, stds = fit_exp_curve(values)
      cvThreshold = self.curve_variance
      a = params[0]
      b = params[1]
      c = params[2]
      if(a < 0 and b < 0):
        cv_a = a/stds[0]
        cv_b = b/stds[1]
        cv_c = c/stds[2]
        if(cv_a < cvThreshold and cv_b < cvThreshold and cv_c < cvThreshold):
          return True
        else:
          return False
      else:
        return False

    #check if fits decreasing exponentials curve (cThreshold close to 0)
    def checkExpCurveDecrease(values):
      params, stds = fit_exp_curve(values)
      cvThreshold = self.curve_variance   
      cThreshold = self.bottom_threshold
      a = params[0]
      b = params[1]
      c = params[2]
      if(a > 0 and b < 0):
        cv_a = a/stds[0]
        cv_b = b/stds[1]
        cv_c = c/stds[2]
        if(cv_a < cvThreshold and cv_b < cvThreshold and cv_c < cvThreshold):
          return True
        else:
          return False
      else:
        return False

    #check if it fits a log curve
    def checkLogCurveIncrease(values):
      params, stds = fit_log_curve(values)
      cvThreshold = self.curve_variance
      a = params[0]
      b = params[1]
      c = params[2]
      if(a > 0 and b > 0):
        cv_a = a/stds[0]
        cv_b = b/stds[1]
        cv_c = c/stds[2]
        if(cv_a < cvThreshold and cv_b < cvThreshold and cv_c < cvThreshold):
          return True
        else:
          return False
      else:
        return False

    #Scenario 3 (Special case of increasing exponential, cThreshold close to 1)
    def checkAccuracyIncrease(values):
      params, stds = fit_log_curve(values)
      cvThreshold = self.curve_variance   
      cThreshold = self.top_threshold
      a = params[0]
      b = params[1]
      c = params[2]
      if(a < 0 and b < 0 and c > cThreshold):
        cv_a = a/stds[0]
        cv_b = b/stds[1]
        cv_c = c/stds[2]
        if(cv_a < cvThreshold and cv_b < cvThreshold and cv_c < cvThreshold):
          return True
        else:
          return False
      else:
        return False
    

    def check_convergence_failure(self):
      d_loss_1_decrease = checkDecreasingLoss(self.data["disc_loss_1"])
      if(d_loss_1_decrease):
        print("POSSIBLE CONVERGENCE FAILURE: Disciminator Loss (#1) is too low")

      d_loss_2_decrease = checkDecreasingLoss(self.data["disc_loss_2"])
      if(d_loss_2_decrease):
        print("POSSIBLE CONVERGENCE FAILURE: Disciminator Loss (#2) is too low")

      g_loss_decrease = checkDecreasingLoss(self.data["gen_loss"])
      if(g_loss_decrease):
        print("POSSIBLE CONVERGENCE FAILURE: Generator Loss is too low")
        
      g_loss_increase_exp = checkExpCurveIncrease(self.data["gen_loss"])
      if(g_loss_increase_exp):
        print("POSSIBLE CONVERGENCE FAILURE: Generator Loss is continuing to increasing")

      g_loss_increase_log = checkLogCurveIncrease(self.data["gen_loss"]) 
      if(g_loss_increase_log):
        print("POSSIBLE CONVERGENCE FAILURE: Generator Loss is continuing to increasing")

      acc_1_increase = checkAccuracyIncrease(self.data["acc_1"])
      if(acc_1_increase):
        print("POSSIBLE CONVERGENCE FAILURE: Disciminator Accuracy (#1) is near 100%")

      acc_2_increase = checkAccuracyIncrease(self.data["acc_2"])
      if(acc_2_increase):
        print("POSSIBLE CONVERGENCE FAILURE: Disciminator Accuracy (#2) is near 100%")
          


