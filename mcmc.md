---
marp: true
theme: experiment
math: mathjax
paginate: true
---
# Markov Chain Monte Carlo: Fancy Guess and Check!
<style scoped>{text-align: center}</style>

<!-- ---
## Marp
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
</style>
* We learned all about Markdown when we looked at Jupyter Lab. Well, Marp is one way of      making slide presentations using Markdown!

* <span style="color:#57ABDC;">"But, why not use $\mathrm{\LaTeX}$? Is Markdown better?"</span>
* <img src=Well_Yes_But_Actually_No.jpg style="width:35%;height:auto;" class="center_small">
* Generally speaking, $\mathrm{\LaTeX}$ Beamer is much better for slides for numerous reasons. While you can include animations in those slides, *and* the PDF specification does support animations, they are not widely support.

* Marp can export slides to HTML, so animations can be included as videos that should work in any web browser! -->

---
## Words of caution...
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* This lecture will gloss over *a lot* of details.

* Like a *whole lot*

* Why?

* Because we don't have time to fit in a full course on statistics and then another full course on Bayesian statistics.

* What I hope the take away to be...
  * A basic idea of what things are
  * A basic idea of how to use the things
  * A general awareness of different ways of fitting models to data
* If you need in the future, do some further reading on all these topics.


---
## Curve Fitting
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* We have already learned that we can use the SciPy library to fit a curve to our data, why do we need anything else?
    * Sometimes non-linear least-squares doesn't converge
    * Sometimes you have very noisy data
    * Sometimes you have a large parameter space
    * Sometimes you want more information about the fit
* Basically, it's the whole "right tool for the right job" thing again.

* Also, MCMC is quite popular these days in certain areas, so it's good to know what it is, how it works, and how you can use it to fit a model to your own data!

* It's not hard to write your own algorithms to do this, but luckily this is Python!

---
## The Data
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* Before we can fit anything to anything we need the thing to fit too...

* What do you call it?

* Oh yeah! The Data! Let's hop over to Jupyter Lab and make some fake data!

* <img src=Code/data.png class="center" style="width:60%;height:auto;">

---
## Covariance Matrix
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* First, we can note that the *variance* is the *standard deviation* squared, i.e. the variance is $\sigma^{2}$.

* The *covariance matrix* is a square matrix where the diagonal elements are the variance of each data point, and the off-diagonal elements encode information about the *correlation* of the data points.

* For a set of $N$ data-points, the elements of the *sample* covariance matrix are calculated as
* $$
    C_{ij} = \dfrac{1}{N_{s} - 1} \sum_{s=1}^{N_{s}}(y_{i,s} - \mu_{i})(y_{j,s} - \mu_{j})
  $$
* where $N_{s}$ is the number of samples. Note that when $i = j$, we get the formula for the variance.

* When $i \neq j$, we get larger values when the two data values differ more from the mean (or expectation) value.

* If both are under/over the mean, the element is *positive* (correlated). If one is over and the other is under the element is *negative* (anticorrelated).

---
## But we only have the one set of data?
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* Since we made fake data, we only have the one set of fake data. So, what do we do?

* Well, we could generate a lot of fake data and then find the sample covariance, or we could just make a diagonal matrix with the variances on the diagonal.

* From here, we can use SciPy's `curve_fit`
* <img src=Code/least_squares.png style="width:60%;height:auto;" class="center">

---
## What's the likelihood of that?
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* When our errors are not Gaussian and independent, the least-squares method doesn't do very well. So, what do we do now?

* Many observations are well described as being drawn from a multivariate Gaussian distribution with an *inverse* covariance matrix $\mathsf{C}^{-1}$.

* In that case, the *likelihood* of a set of parameters, $\theta$, being the best-fitting is
* $$
    \mathcal{L}(x | \theta, \mathsf{C}^{-1}) = \dfrac{|\mathsf{C}^{-1}|}{\sqrt{2\pi}}\exp\left[-\dfrac{1}{2}\chi^{2}(x, \theta, \mathsf{C}^{-1})\right],
  $$
* where
  $$
    \chi^{2}(x, \theta, \mathsf{C}^{-1})\equiv \sum_{ij}[x_{i}^{d} - x_{i}(\theta)](\mathsf{C}^{-1})_{ij}[x_{j}^{d} - x_{j}(\theta)].
  $$
* where $x^{d}$ are the data and $x(\theta)$ are the model values.

* We can now mathematically find the parameters $\theta$ that *maximize the likelihood*.

---
## Logarithms
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* Generally, we don't work with the likelihood directly. Instead, it's often the *log-likelihood* that we care about.
* $$
    \ln\mathcal{L} = \underbrace{\ln\left(\dfrac{|\mathsf{C}^{-1}|}{2\pi}\right)}_{\mathsf{\text{constant}}} - \dfrac{1}{2}\chi^{2}.
  $$
* Now, when we made our data we had this parameter $f$ which was basically saying that we are *under estimating* the error's in our data by that fraction. In that case, the correct log-likelihood is
* $$
    \ln\mathcal{L} = -\dfrac{1}{2}\sum_{n}\left[\dfrac{(y_{n} - mx_{n} - b)^{2}}{s_{n}^{2}} + \ln(2\pi s_{n}^{2})\right]
  $$
* where
  $$
    s_{n}^{2} = \sigma_{n}^{2} + f^{2}(mx_{n} + b)^{2},
  $$
  by construction.


---
## SciPy Minimize
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* Now, we just realize that minimizing the *negative* log-likelihood is the same as maximizing the likelihood (i.e. we are minimizing $\chi^{2}$).
* Since we haven't talked about it specifically, and it is a function that comes in handy (i.e. minimizing the $\chi^{2}$ value), let's take a closer look at the function signature like we usually do
* ```python
    minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, 
             constraints=(), tol=None, callback=None, options=None)
  ```
  * `fun`: The function to be minimized
  * `x0`: Initial guess at "where" the minimum is
  * `args`: Things to pass to `fun` so that it works
  * `bounds`: Bounds on variables to prevent things from running away
* For the other parameters here, consult the documentation but you probably won't need to use them. For example, we will just need to pass the function, initial guess, and arguments to pass to our function. The defaults for everything else are fine.

---
## Find the maximum likelihood with the minimize function!
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* So, we just need to define our negative log-likelihood function, and then use the `minimize` function from SciPy to find the optimal values of our parameter!

* Back to Jupyter Lab!

* <img src=Code/max_likelihood.png style="width:60%;height:auto;" class="center">

---
## What about the errors on the parameters!?
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* When we used least-squares, we automatically got some error estimates on the parameter values. We don't get that with the maximum likelihood method.

* We can also note that we did get values that more closely matched the truth from the maximum likelihood, so it is clearly better in some cases.

* We can get estimates on the errors of our parameters *as well as* correlations between them using Markov Chain Monte Carlo.

* <span style="color:#57ABDC;">"You've said that *a lot* already. Just tell me what it is!"</span>

* We are getting there right now!

* When we say *Markov Chain Monte Carlo* we are actually referring to a *class* of algorithms that draws samples from a probability distribution.

* The terminology gets dense quick... remember this is just so you have heard these things, we don't need to fully understand all the terms to understand the process.

---
## Metropolis-Hastings
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* The simplest example of an MCMC algorithm is that of Metropolis-Hastings.

* The method basically says
    * Calculate the likelihood at your initial guess of the model parameters
    * Randomly select some point in parameter space from some box around that initial guess
    * Calculate the likelihood at this random point, and then the likelihood ratio $\mathcal{L}_{1}/\mathcal{L}_{0}$.
    * Select a uniform random number between 0 and 1, call it $r$
    * If $\mathcal{L}_{1}/\mathcal{L}_{0} > r$, move to the randomly selected point in parameter space.
    * Else, stay at the "current" point.
    * Repeat the process *a lot* of times, recording the "accepted" parameter values at each step even if they are the same as the previous step.
    * Calculate the covariance matrix of the parameters from the accepted parameter values.
* The average parameter values of all the steps *should* converge towards the maximum likelihood.

---
## What does it look like?
<video src=metropolis_hastings_start.mp4 class="center_big" autoplay muted>

---

## What does it look like if more and faster!?
<video src=metropolis_hastings_full.mp4 class="center_big" autoplay muted>

---
## More vocabulary
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* What we are doing is exploring the *posterior* distribution. Basically, we a exploring how the likelihood behaves and trying to find its shape.

* To do this we need to find a good balance between the *chain* moving to new locations and exploring parameter space efficiently.

* If the random draw around the current point is too small, likelihood is very similar, very likely to move, moves around a lot in a small space.

* If the random draw around the current point is too big, likelihood is very different, unlikely to move, does not explore efficiently.

* Want an acceptence ratio of about 0.234 for most chains. Does somewhat depend on number of parameters.

* We can also place *priors* on the parameter values based on previous knowledge.

---
## So, how do I do this?
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* As mentioned, we can write our own algorithm. But, it is almost always better to use a library that is vetted.

* For this we will use the emcee library
  ```bash
  pip install emcee
  ```
  ```bash
  sudo pamac install python-emcee
  ```
* We will also need the corner library to make the usual corner plots associated with MCMC parameter estimation
  ```bash
  pip install corner
  ```
  ```bash
  sudo pamac install python-corner
  ```

---
## And then?
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* Then we can set our priors by making a function
* ```python
  def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf
  ```
* This places hard limits on the values our parameters can take.

* Then we can include those priors so that the likelihood drops to zero if the parameters try to wander out of bounds
* ```python
  def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
  ```

---
## And then?
<style scoped>ul {margin-left: 0px; padding-left: 0px; list-style-type: none}
    ul li ul {margin-left: 60px; padding-left: auto; list-style-type: disc}
</style>
* Define the log likelihood
  ```python
  def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta # Unpack the parameters
    model = m*x + b
    sigma2 = yerr**2 + model**2*np.exp(2*log_f)
    return -0.5*np.sum((y - model)**2/sigma2 + np.log(sigma2))
  ```
* Then run the chain!
  ```python
  import emcee

  pos = soln.x + 1e-4 * np.random.randn(32, 3)
  nwalkers, ndim = pos.shape

  sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x, y, yerr)
  )

  sampler.run_mcmc(pos, 10000, progress=True);
  ```
* Now, lets pop back over to Jupyter lab!

---
## But a few more things!
After class I did add a few more things to the Jupyter Notebook that we were working in that shows how to "marginalize" over the other parameters and how to get the errors to add to our maximum likelihood estimation. So, make sure to take a look at that!