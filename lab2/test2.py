import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider


# The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)


def damping(t, x0, hl):
    d = 2 * np.log(2) / hl
    return (x0 + d * x0 * t) * np.exp(-d * t)


def co_damping(t, x0, hl):
    return damping(t, 0.5 * x0, hl) + damping(2 - t, -0.5 * x0, hl)


t = np.linspace(0, 2, 2000)

# Define initial parameters
init_amplitude = 5
init_frequency = 3
init_target = 1
init_halflife = 0.2

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
# line, = ax.plot(t, f(t, init_amplitude, init_frequency), lw=2)
line, = ax.plot(t, co_damping(t, init_target, init_halflife), lw=2)
ax.set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Frequency [Hz]',
#     valmin=0.1,
#     valmax=30,
#     valinit=init_frequency,
# )

hl_slider = Slider(
    ax=axfreq,
    label='Half Life [s]',
    valmin=0.01,
    valmax=0.5,
    valinit=init_halflife,
)

# Make a vertically oriented slider to control the amplitude
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
# amp_slider = Slider(
#     ax=axamp,
#     label="Amplitude",
#     valmin=0,
#     valmax=10,
#     valinit=init_amplitude,
#     orientation="vertical"
# )

tgt_slider = Slider(
    ax=axamp,
    label="Target",
    valmin=0.1,
    valmax=10,
    valinit=init_target,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
# def update(val):
#     line.set_ydata(f(t, amp_slider.val, freq_slider.val))
#     fig.canvas.draw_idle()


def update(val):
    line.set_ydata(co_damping(t, tgt_slider.val, hl_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
# freq_slider.on_changed(update)
# amp_slider.on_changed(update)

hl_slider.on_changed(update)
tgt_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


# def reset(event):
#     freq_slider.reset()
#     amp_slider.reset()


def reset(event):
    hl_slider.reset()
    tgt_slider.reset()


button.on_clicked(reset)

plt.show()
