"""
Main code authors:
Colin Simpson, Ben Whittington

Using code written by:
David Dempsey, Andrew Mason
"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import VBox, HBox, interactive_output, fixed


def demo_methods():
	"""
	Create widgets, set layout, and use as inputs to run the main code
	"""

	steps = widgets.IntSlider(
		value=3,
		description='Steps',
		min=0,
		max=50,
		continuous_update=False,
	)

	h = widgets.FloatSlider(
		value=0.5,
		description='Step Size',
		min=0.1,
		max=1,
		step=0.1,
	)

	y0 = widgets.FloatSlider(
		value=0.,
		description='y(0)',
		min=-1.,
		max=1.,
		step=0.1,
	)

	method = widgets.Dropdown(
		options=['Euler', 'IEuler', 'ClassicRK4'],
		value='ClassicRK4',
		description='Solution Method'
	)

	eqn = widgets.Dropdown(
		options=["Function1", "Function2", "Function3", "Function4"],
		description='Derivative Function'
	)

	zoom = widgets.Checkbox(
		value=True,
		description='Zoom In'
	)

	deriv = widgets.Checkbox(
		value=False,
		description='Show Derivative Field'
	)

	evals = widgets.Checkbox(
		value=True,
		description='Show Next Step'
	)

	neval = widgets.IntSlider(
		value=4,
		description='Number of Derivative Evaluations',
		min=0,
		max=4,
		continuous_update=False
	)

	# Update value of number of evaluations widget based on method widget
	def change_method(*args):
		if method.value == 'Euler':
			neval.min = 0
			neval.max = 0
			neval.value = 0
		elif method.value == 'IEuler':
			neval.min = 0
			neval.max = 2
			neval.value = 0
		elif method.value == 'ClassicRK4':
			neval.min = 0
			neval.max = 4
			neval.value = 0
	method.observe(change_method)

	# create list of controls to pass to function as inputs
	controls = {
		'h': h,
		'steps': steps,
		'y0': y0,
		'neval': neval,
		'method': method,
		'eqn': eqn,
		'zoom': zoom,
		'show_deriv': deriv,
		'show_grad': evals,
		'rk2': fixed(1.)
	}

	# set up the layout for UI interface
	col1 = [h, steps, y0]
	col2 = [eqn, method, neval]
	col3 = [zoom, evals, deriv]
	layout = HBox([VBox(col1), VBox(col2), VBox(col3)])

	return VBox([layout, interactive_output(run_ode_methods, controls)])


def demo_rk2():
	"""
	Create widgets, set layout, and use as inputs to run the main code
	"""

	steps = widgets.IntSlider(
		value=3,
		description='Steps',
		min=0,
		max=50,
		continuous_update=False,
	)

	h = widgets.FloatSlider(
		value=0.5,
		description='Step Size',
		min=0.1,
		max=1,
		step=0.1,
	)

	y0 = widgets.FloatSlider(
		value=0.,
		description='y(0)',
		min=-1.,
		max=1.,
		step=0.1,
	)

	eqn = widgets.Dropdown(
		options=["Function1", "Function2", "Function3", "Function4"],
		description='Derivative Function'
	)

	rk2 = widgets.FloatSlider(
		value=0.5,
		min=0.1,
		max=1.0,
		step=0.1,
		description='Beta for RK2',
		continuous_update=True
	)

	zoom = widgets.Checkbox(
		value=True,
		description='Zoom In'
	)

	deriv = widgets.Checkbox(
		value=False,
		description='Show Derivative Field'
	)

	evals = widgets.Checkbox(
		value=True,
		description='Show Next Step'
	)

	# create list of controls to pass to function as inputs
	controls = {
		'h': h,
		'steps': steps,
		'y0': y0,
		'neval': fixed(2),
		'method': fixed("RK2"),
		'eqn': eqn,
		'zoom': zoom,
		'show_deriv': deriv,
		'show_grad': evals,
		'rk2': rk2
	}

	# set up the layout for UI interface
	col1 = [h, steps, y0]
	col2 = [eqn, rk2]
	col3 = [zoom, evals, deriv]
	layout = HBox([VBox(col1), VBox(col2), VBox(col3)])

	return VBox([layout, interactive_output(run_ode_methods, controls)])


def demo_error():

	nstep = widgets.IntRangeSlider(
		value=[10, 40],
		min=1,
		max=50,
		step=1,
		description='Number Steps',
		continuous_update=False
	)

	t_end = widgets.FloatSlider(
		value=5.0,
		min=1.0,
		max=10.0,
		step=0.5,
		description='Time End',
		continuous_update=False
	)

	eqnkey = widgets.Dropdown(
		options=["Function1", "Function2"],
		description='Derivative Function'
	)

	method_e = widgets.Checkbox(
		value=True,
		description='Euler Method'
	)

	method_ie = widgets.Checkbox(
		value=False,
		description='Improved Euler Method'
	)

	method_rk = widgets.Checkbox(
		value=False,
		description='RK4 Method'
	)

	# create list of controls to pass to function
	controls = {
		'nstep': nstep,
		't_end': t_end,
		'y0': fixed(0.),
		'eqn': eqnkey,
		'euler': method_e,
		'ieuler': method_ie,
		'rk4': method_rk
		# 'method_ie': method_ie,
		# 'method_rk': method_rk
	}

	# setup layout for UI interface
	col1 = [nstep, t_end]
	col2 = [eqnkey]
	col3 = [method_e, method_ie, method_rk]
	layout = HBox([VBox(col1), VBox(col2), VBox(col3)])

	return VBox([layout, interactive_output(run_error, controls)])


def run_ode_methods(h, steps, y0, neval, eqn, method, zoom, show_deriv, show_grad, rk2):
	"""
	Solves and plots a first-order ODE.

	:param h: step size.
	:param steps: number of steps.
	:param y0: initial condition.
	:param neval: number of derivative evaluations.
	:param eqn: equation name.
	:param method: solution method.
	:param zoom: set True to zoom in plot window.
	:param show_deriv: set True to show next step.
	:param show_grad: set True to show derivative scalar field.
	:param rk2: value of beta and gamma in a RK2 scheme.

	:return:
	No return.
	"""
	# set up an equation dictionary and evaluate for chosen equation
	dict_equation = {
		"Function1": [0, 0, 5, 0, 1.2],
		"Function2": [1, 0, 10, -1.5, 1.5],
		"Function3": [2, 0, 5, 0, 1.2],
		"Function4": [3, 0, 10, -1.5, 1.5],
	}
	function_number, x_min, x_max, y_min, y_max = dict_equation[eqn]

	# set up a solution method dictionary
	dict_method = {
		"Euler": [[1.], [0.], [0.]],
		"IEuler": [[1., 1.], [0., 1.], [0., 1.]],
		"RK2": [[1-0.5/rk2, 0.5/rk2], [0., rk2], [0., rk2]],
		"ClassicRK4": [[1., 2., 2., 1.], [0., 0.5, 0.5, 1.], [0., 0.5, 0.5, 1.]],
	}
	alpha, beta, gamma = dict_method[method]

	# set upper bound for solution
	x_end = np.min([h*steps, x_max])

	# evaluate and plot solution to ODE up to current step number
	x, y = solver(function_number, 0, y0, x_end, h, alpha, beta, gamma)

	# create the figure window
	plt.figure(figsize=(16, 8))
	ax = plt.axes()

	# set some plotting related variables
	scale = 0.1
	buffer =  5.
	colors = ['b', 'r', 'g', 'm', 'c']

	# plot the numerical solution
	ax.plot(x, y, marker='o', color='k', label='Numerical Solution')

	# calculate one more solution step and save locations of derivative evaluations
	y_next, eval_x, eval_y, eval_f = step(function_number, x[-1], y[-1], h, alpha, beta, gamma)
	if method == "RK2":
		print('Alpha ', alpha, '\nBeta  ', beta, '\nGamma ', gamma)

	# recalculate axes limits if zoom requested
	if zoom:
		x_values = np.array([x[-1] - scale * h, x[-1] + h + scale * h])
		y_values = np.array([y[-1], y_next])
		for i in range(len(eval_f)):
			y_values = np.append(y_values, eval_y[i] - scale * h * eval_f[i])
			y_values = np.append(y_values, eval_y[i] + scale * h * eval_f[i])
		x_min = np.min(x_values) - (np.max(x_values) - np.min(x_values)) / buffer
		x_max = np.max(x_values) + (np.max(x_values) - np.min(x_values)) / buffer
		y_min = np.min(y_values) - (np.max(y_values) - np.min(y_values)) / buffer
		y_max = np.max(y_values) + (np.max(y_values) - np.min(y_values)) / buffer

	# plot the set number of derivative(s) evaluation(s)
	if show_grad:

		# plot the line and marker for actual step taken,
		ax.plot(
			[x[-1], x[-1] + h],
			[y[-1], y_next],
			color='k', ls='-', lw=2., marker='s', label="Next Step"
		)

		# for each new derivative evaluation, plot some related stuff
		for i in range(neval):

			# small solid line showing local derivative
			ax.plot(
				[eval_x[i] - scale * h, eval_x[i] + scale * h],
				[eval_y[i] - scale * h * eval_f[i], eval_y[i] + scale * h * eval_f[i]],
				ls='-', color=colors[i],  label=r"$f_{%d}$" % (i,)
			)

			# marker where derivatives are evaluated
			if i > 0:
				ax.plot(eval_x[i], eval_y[i], color=colors[i], marker='s', ls='')

			# dashed line showing where next derivative evaluation is taken
			if i < neval-1:
				ax.plot(
					[eval_x[0], eval_x[i+1]],
					[eval_y[0], eval_y[0] + (eval_x[i+1] - eval_x[0]) * eval_f[i]],
					ls='--', color=colors[i]
				)

	# plot the derivative scalar function as quiver plot - modified code from AJM
	if show_deriv:

		# set some quiver related values
		quiver_step = 0.05
		density = 25

		# create the 2d mesh grid based on plot window limits
		x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, density), np.linspace(y_min, y_max, density))

		u = np.full(x_grid.shape, quiver_step/2)
		v = np.full(y_grid.shape, 0.08)
		for i in range(u.shape[0]):
			for j in range(v.shape[1]):
				xq = x_grid[i, j]
				yq = y_grid[i, j]
				v[i, j] = calculate_derivative(function_number, xq, yq) * u[i, j]

		# plot the small quivers indicating local derivative
		ax.quiver(
			x_grid, y_grid, u, v, units="xy", headlength=0, headaxislength=0, headwidth=0,
			angles='xy', scale_units='xy', pivot='mid', alpha=0.3
		)

	# set the default axes limits
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)

	# set plot labels
	ax.set_xlabel('t')
	ax.set_ylabel('y(t)')

	# set legend
	ax.legend()

	# show plot in notebook
	plt.show()


def run_error(nstep, t_end, y0, eqn, euler, ieuler, rk4):

	# set up the equation and parameters for each case
	equation = {
		"Function1": [0, 0, t_end, 0, 1.2],
		"Function2": [1, 0, t_end, -1.5, 1.5]
	}
	function_number, x_min, x_max, y_min, y_max = equation[eqn]

	# set up the equation and parameters for each case
	dict_method = {
		"Euler": [[1.], [0.], [0.]],
		"IEuler": [[1., 1.], [0., 1.], [0., 1.]],
		"ClassicRK4": [[1., 2., 2., 1.], [0., 0.5, 0.5, 1.], [0., 0.5, 0.5, 1.]],
	}

	# set list of method booleans
	methods = [euler, ieuler, rk4]

	# set step sizes for which we will solve the ODE
	h = [t_end/i for i in np.arange(nstep[0], nstep[1]+1, 1)]

	# create the figure window
	plt.figure(figsize=(16, 8))
	ax = plt.axes()

	# set colours for each method
	colors = ['b', 'r', 'g']

	for m in range(len(methods)):

		# choose method to use
		if m == 0 and methods[m]:
			method = "Euler"
		elif m == 1 and methods[m]:
			method = "IEuler"
		elif m == 2 and methods[m]:
			method = "ClassicRK4"
		else:
			method = None

		if methods[m]:
			# set rk parameters for current model
			alpha, beta, gamma = dict_method[method]

			for i in range(len(h)):

				# solve ode for current step size
				x, y = solver(function_number, 0., y0, t_end, h[i], alpha, beta, gamma)

				# absolute error
				# error = np.abs(calculate_analytic(function_number, x)[-1] - y[-1])

				# mean absolute error
				error = sum(abs(calculate_analytic(function_number, x)[:len(y)] - y))/len(x)

				# set label only for one of the steps
				if i == 0:
					label = method
				else:
					label = None

				# plot the error for the current step
				ax.plot(1./h[i], error, marker='s', color=colors[m], label=label)

	# set labels and legend
	ax.set_xlabel('1/h')
	ax.set_ylabel('Mean absolute error')
	ax.legend()

	# plot the figure on the notebook
	plt.show()


def calculate_derivative(eqn, x, y):
	if eqn == 0:
		derivative = -1. * y + 1. + 0. * x
	elif eqn == 1:
		derivative = np.cos(x) + 0. * y
	elif eqn == 2:
		derivative = x * np.exp(np.sin(x)) + 0. * y
	elif eqn == 3:
		derivative = x + y
	else:
		derivative = None
	return derivative


def calculate_analytic(eqn, x):
	# requires an initial condition of y(x=0)=0
	if eqn == 0:
		analytic = 1. - np.exp(-1. * x)
	elif eqn == 1:
		analytic = np.sin(x)
	else:
		analytic = None
	return analytic


def solver(function_number, x0, y0, x1, h, alpha, beta, gamma):

	# calculate the number of steps required
	nx = int(np.ceil((x1 - x0) / h))

	# calculate the array of where we will solve for the independent variable
	xs = x0 + np.arange(nx + 1) * h

	# initialise the solution array
	ys = np.zeros(len(xs))

	# set initial condition of solution array
	ys[0] = y0

	# calculate the solution for each step
	for i in range(len(xs) - 1):
		ys[i+1], _, _, _ = step(function_number, xs[i], ys[i], h, alpha, beta, gamma)

	return xs, ys


def step(function_number, xk, yk, h, alpha, beta, gamma):

	# initialise derivative
	derivative = 0.

	# initialise weighted average derivative required for an RK method
	average = 0.

	# initialise lists for derivative evaluation information
	eval_x, eval_y, eval_f = [], [], []

	# evaluate each derivative
	for i in range(len(alpha)):

		# append where the next derivative evaluation is taken
		eval_x.append(xk + beta[i] * h)
		eval_y.append(yk + gamma[i] * h * derivative)

		# calculate the next derivative value
		derivative = calculate_derivative(function_number, xk + beta[i] * h, yk + gamma[i] * h * derivative)

		# append to list of derivatives calculated
		eval_f.append(derivative)

		# add most recent derivative to the running average
		average += derivative * alpha[i]

	# calculate the weighted average derivative
	average = np.copy(average)/np.sum(alpha)

	# append the end location and weighted average derivative evaluation
	eval_x.append(xk + h)
	eval_y.append(yk + h * average)
	eval_f.append(average)

	# calculate the next solution estimate
	yk1 = yk + h * average

	return yk1, eval_x, eval_y, eval_f
