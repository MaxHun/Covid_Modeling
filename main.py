"""
1) 

We took Bayern, Berlin, and Brandenburg from the list and considered the
first 365 days of the time axis. 

2)

The model consists of 3 compartments, the susceptible group S, the 
infected I and the recovered R. Every person in I has beta many
critical contacts every day as long as they are infected. A critial 
contact leads to an infection only if the other person is susceptible.
Every person stays infected for D days and then joins the recovered 
compartment. We assume that nobody dies of the desease and that the 
population number is constant.

The system is governed by the following equations:

dS/dt = -beta*I * S/N
  (every infected person has -b critical contacts, S/N of which are with
   a susceptible person that will get infected)
   
dI/dt = beta*I * S/N - gamma*I
  (first term comes from the people that get infected, second from
   the people that recover after D=1/gamma days)

dR/dt = gamma*I
  (the recovered people from above)


3)

The data consists of the number of cases that were reported within the 
last 7 days. Assuming that D=7 and that all cases are reported right
at the start of the contagious phase (which is of course a dramatic 
simplification of reality), this is exactly the number I of infected
people. The 7-days incidence is given by the number of cases within 
the last 7 days per 10,000 people in the region in question. This 
gives access to the population number N that was used. We worked these
numbers out seperately, not to load another sheet with every call of 
main().
The only free parameter is beta. We fit the model to the data
using the Powell method. The fits give the beta values 0.1677, 0.1662, 
and 0.1718 for the three states. This is a realistic value, since the 
R-value, i.e. the number of people infected by one person in total is
given by beta*D, is hence around 1.1 ... 1.2, which coincides with the
values given by RKI. 


4)

We observe that while the general shape of the data is met by the 
model, it does not allow for many details. In particular, the solution 
of the model only has one maximum, while the data suggests the 
existence of more extreme points in the given time interval. As for
Bayern, the first maximum in the data is followed by a sharp decline of
the incidence, which is particularly hard for the model to adapt to 
properly.
Furthermore, we see that the model is very sensitive to small changes 
of the parameter beta, as all states have relatively similar beta values.


5)

One way to improve the result would be to add more compartments to the 
model, such as a vaccinated. This would change the first equation to

dS/dt = -beta*I * S/N - v*S

and add a fourth equation

dV/dt = v*S

with the vaccination rate v. Another way would be to consider the deaths
caused by the desease, adding a term -mu_D*I on the rhs of the second 
equation and adding it to therhs of the third, where mu_D is the 
mortality of the desease and make D another compartment with 

dD/dt = mu_D*I.

Moreover, we could consider an exposed compartment with individuals 
that already have already had a critical contact, but are not contagious
yet, with similar adaptations to the equations. This would lead to more
parameters to be fitted, like the mortality or the incubation time.
In addition, the parameters could be modeled as time-dependent 
quantities. This could reflect measures taken against the outbreak by
effectively reducing the number of people that get infected by one 
individual. For that, one would set beta=beta(t, params) and solve for 
the parameters of the beta function.


6)

To couple n states in one model, the easiest way is to add all the
numbers together, given that the states are already compromised of 
multiple districts. However, this approach would lose all the spacial
resolution. Instead, all the states could be modeled seperately with
people traveling between them.
Mathematically, this means that we have each of the three presented 
equations n times, with all the S, I, and R indexed by the regions.
To model individuals moving between two states A and B, we would 
add these flows by writing 

dS_A/dt = -beta*I_A * S_A/I_A + (S_B/N_B)*N_BA - (S_A/N_A)*N_AB
dI_A/dt = beta*I * S_A/N_A - gamma*I + (I_B/N_B)*N_BA - (I_A/N_A)*N_AB
dR/dt = gamma*I + (I_B/N_B)*N_BA - (I_A/N_A)*N_AB,

and similarly for S_B, I_B, and R_B, where N_AB are the people 
travelling from A to B and N_BA are the people travelling in the other
direction. To reflect a more realistic behaviour, the infected could be
given a smaller probability to travel, which would be realised by means 
of an additional factor in front of the above ratios (S_B/N_B), 
(I_B/N_B), etc.
"""


import openpyxl as op
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np
import lmfit


def Instance(beta, gamma, S_0, I_0, R_0, num_days):
    """
    Instance of our model. Takes the parameters beta and gamma,
    initial values S_0, I_0, R_0, and the number of days to be 
    simulated and returns S, I and R as arrays.
    """
    N = S_0 + I_0 + R_0               # total population
    t = np.arange(num_days) 
    y_0 = S_0, I_0, R_0               # initial values
    
    def deriv(y, t, N, b, g):
        """
        Gives the derivative at point t with y=sol(t) given the point 
        and the parameters.
        """
        S, I, R = y
        dSdt = -b*S*I / N
        dIdt = b*S*I/N - g*I
        dRdt = g*I
        return dSdt, dIdt, dRdt

    sol = odeint(deriv, y_0, t, args=(N, beta, gamma))
    S, I, R = sol.T
    
    return S, I, R

def fitting(x, beta, I_0, N, num_days):
    """
    Takes the parameters and a time array x, returns I(t), t in x.
    """
    # Start with I_0 infected, rest susceptible:
    
    S, I, R = Instance(beta, 1/7, N-I_0, I_0, 0, len(x))
    return I

def main():
    """
    Reads the data and plots the fits.
    """
    f_name = "Fallzahlen_Kum_Tab_aktuell.xlsx"

    wb = op.load_workbook(filename = f_name)

    cases_sheet = wb['BL_7-Tage-Fallzahlen (fixiert)']

    rows = list(cases_sheet.iter_rows(values_only=True))

    f = plt.figure(figsize=(20,10))
    plt.subplot(131)
    for i in range(3):
        row_number = 6+i                          # start in row 6
        I = rows[row_number][1:366]
        state = rows[row_number][0]
        N = {"Bayern" : 13.076721e6, "Berlin" : 3.748148e6,\
            "Brandenburg" : 2.511917e6}[state]    # population

        
        days = rows[4][1:366]                     # first year

        


        # initialise the model, set the params and fit:    
        
        model = lmfit.Model(fitting)
        model.set_param_hint("beta", value=-0., min=0, max=1, vary=True)
        params = model.make_params(I_0=I[0], N=N, num_days=len(I))
        params["N"].vary = False
        params["I_0"].vary = False
        params["num_days"].vary = False
        result = model.fit(np.array(list(I)), params, method="powell",\
                           x=np.arange(len(I)))

        # give the optimal values:
        print(result.values)

        # plotting:
        result.plot_fit(datafmt=".", title=state, ylabel="I", 
                        xlabel="days", ax=plt.subplot(1, 3, i+1))
    plt.show()
    
if __name__ == "__main__":
    print(__doc__)
    main()
    

