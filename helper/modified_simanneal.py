from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import datetime
import math
import pickle
import random
import signal
import sys
import timeit
import time


def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)


class Annealer(object):

    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """

    __metaclass__ = abc.ABCMeta

    # defaults
    Tmax = 2500.0
    Tmin = 2.5
    steps = 50000
    step = 0
    updates = 100
    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False
    eps = 1
    # placeholders
    best_state = None
    best_energy = None
    start = None

    def __init__(self, initial_state=None, load_state=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        elif load_state:
            self.state = self.load_state(load_state)
        else:
            raise ValueError('No valid values supplied for neither \
            initial_state nor load_state')

        signal.signal(signal.SIGINT, self.set_user_exit)

    def save_state(self, fname=None):
        """Saves state to pickle"""
        if not fname:
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = date + "_energy_" + str(self.energy()) + ".state"
        with open(fname, "wb") as fh:
            pickle.dump(self.state, fh)

    def load_state(self, fname=None):
        """Loads state from pickle"""
        with open(fname, 'rb') as fh:
            self.state = pickle.load(fh)

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        pass

    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass

    def set_user_exit(self, signum, frame):
        """Raises the user_exit flag, further iterations are stopped
        """
        self.user_exit = True

    def set_schedule(self, schedule):
        """Takes the output from `auto` and sets the attributes
        """
        self.Tmax = schedule['tmax']
        self.Tmin = schedule['tmin']
        self.steps = int(schedule['steps'])
        self.updates = int(schedule['updates'])

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of
        * deepcopy : use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()

    def update(self, *args, **kwargs):
        """Wrapper for internal update.
        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E, acceptance, improvement):
        """Default update, outputs to stderr.
        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time, and remaining time.
        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.
        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible."""

        # elapsed = time.time() - self.start
        # if step == 0:
        #     print(' Temperature        Energy    Accept   Improve     Elapsed   Remaining',file=sys.stderr)
        #     print('\r%12.5f  %12.2f                      %s            ' %
        #           (T, E, time_string(elapsed)), file=sys.stderr, end="\r")
        #     sys.stderr.flush()
        # else:
        #     remain = (self.steps - step) * (elapsed / step)
        #     print('\r%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s  %s\r' %
        #           (T, E, 100.0 * acceptance, 100.0 * improvement,
        #            time_string(elapsed), time_string(remain)), file=sys.stderr, end="\r")
        #     sys.stderr.flush()

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.
        Parameters
        state : an initial arrangement of the system
        Returns
        (state, energy): the best state and energy found.
        """
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(self.step, T, E, None, None)
        start_time = timeit.default_timer()
        self.step = 0
        while  T > self.eps:
            self.step = 0
            print T
            for i in xrange(0,self.steps):
                self.step += 1
                self.move()
                E = self.energy()
                dE = E - prevEnergy
                trials += 1
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    E = prevEnergy
                else:
                    # Accept new state and compare to best state
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
                    if E < self.best_energy:
                        self.best_state = self.copy_state(self.state)
                        self.best_energy = E
                # if self.updates > 1:
                #     if (self.step // updateWavelength) > ((self.step - 1) // updateWavelength):
                #         self.update(
                #             self.step, T, E, accepts / trials, improves / trials)
                #         trials, accepts, improves = 0, 0, 0
            T = T*0.90

        # # Attempt moves to new states
        # start_time = timeit.default_timer()
        # self.step = 0
        # while self.step < self.steps and not self.user_exit:
        #     # if self.step%100 == 0:
        #     #     end_time = timeit.default_timer()
        #     #     print self.step,end_time - start_time
        #     #     start_time = end_time
        #     self.step += 1
        #     T = self.Tmax * math.exp(Tfactor * self.step / self.steps)
        #     self.move()
        #     E = self.energy()
        #     dE = E - prevEnergy
        #     trials += 1
        #     if dE > 0.0 and math.exp(-dE / T) < random.random():
        #         # Restore previous state
        #         self.state = self.copy_state(prevState)
        #         E = prevEnergy
        #     else:
        #         # Accept new state and compare to best state
        #         accepts += 1
        #         if dE < 0.0:
        #             improves += 1
        #         prevState = self.copy_state(self.state)
        #         prevEnergy = E
        #         if E < self.best_energy:
        #             self.best_state = self.copy_state(self.state)
        #             self.best_energy = E
        #     # if self.updates > 1:
        #     #     if (step // updateWavelength) > ((step - 1) // updateWavelength):
        #     #         self.update(
        #     #             step, T, E, accepts / trials, improves / trials)
        #     #         trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy