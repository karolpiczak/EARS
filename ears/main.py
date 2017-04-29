# -*- coding: utf-8 -*-

import json
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Button, Div, FixedTicker, FuncTickFormatter, HoverTool, Slider
from bokeh.models.callbacks import CustomJS
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.plotting import ColumnDataSource, figure

import audio
from config import *


with open('ears/model_labels.json', 'r') as labels_file:
    labels = json.load(labels_file)

PALETTE = Viridis256
PALETTE_DEFAULT_THRESHOLD = 0.5
SPEC_PALETTE = Viridis256

WIDTHS = [500, 100]
HEIGHTS = [200, 50 + 11 * len(labels)]
GRID_COLOR = '#eeeeee'
TEXT_COLOR = '#555555'
TEXT_FONT = 'Signika'
SLIDER_WIDTH = 250

threshold = Slider(start=0.0, end=1.0, value=PALETTE_DEFAULT_THRESHOLD, step=0.05,
                   callback_policy='mouseup', title='Colorize probabilities greater than',
                   width=SLIDER_WIDTH, css_classes=['threshold-slider'])

SPEC_WIDTH = np.shape(audio.spectrogram)[1]
SPEC_HEIGHT = np.shape(audio.spectrogram)[0]

DETECTION = ColumnDataSource(
    data=dict(
        pos=[],
        label=[],
        value=[],
        pretty_value=[],
        color=[],
    )
)

HISTORY = ColumnDataSource(
    data=dict(
        label=[],
        x=[],
        y=[],
        value=[],
        pretty_value=[],
        color=[],
    ),
)

THRESHOLD = ColumnDataSource(
    data=dict(
        threshold=[PALETTE_DEFAULT_THRESHOLD - 0.01]
    )
)

SPECTROGRAM = ColumnDataSource(
    data=dict(
        value=[np.zeros((SPEC_HEIGHT, SPEC_WIDTH), dtype='float32')]
    )
)

LIVE_AUDIO = ColumnDataSource(
    data=dict(
        signal=[],
    )
)


def to_percentage(number):
    """Convert a float fraction into percentage representation.
    E.g. 0.953212 -> 95.32
    """
    return int(number * 1000) / 10.


def colorizer(value, reverse=False, low=0., high=1., threshold=None):
    if threshold is not None and value < threshold:
        return '#ffffff'
    idx = (value - low) / (high - low) * len(PALETTE)
    idx = int(np.clip(idx, 0, len(PALETTE)))
    return PALETTE[idx] if not reverse else PALETTE[-(idx + 1)]


def plot_spectrogram():
    # Spectrogram image
    plt = figure(plot_width=WIDTHS[0], plot_height=HEIGHTS[0],
                 toolbar_location=None, tools="",
                 x_range=[0, SPEC_WIDTH],
                 y_range=[0, SPEC_HEIGHT])

    plt.image('value', x=0, y=0, dw=SPEC_WIDTH, dh=SPEC_HEIGHT, name='spectrogram',
              color_mapper=LinearColorMapper(SPEC_PALETTE, low=0, high=100), source=SPECTROGRAM)

    # X ticks
    plt.xaxis[0].ticker = FixedTicker(ticks=[])

    # X axis
    plt.xaxis.axis_line_color = None

    # Y ticks
    plt.yaxis[0].ticker = FixedTicker(ticks=[])
    plt.yaxis.major_label_text_font_size = '0pt'
    plt.yaxis.major_tick_line_color = None

    # Y axis
    plt.yaxis.axis_line_color = None
    plt.yaxis.axis_label = 'Mel bands'
    plt.yaxis.axis_label_text_font = TEXT_FONT
    plt.yaxis.axis_label_text_font_size = '8pt'
    plt.yaxis.axis_label_text_font_style = 'normal'

    # Plot fill/border
    plt.background_fill_color = GRID_COLOR
    plt.outline_line_color = GRID_COLOR
    plt.min_border = 10

    # Plot title
    plt.title.text = 'Mel-scaled power spectrogram (perceptually weighted):'
    plt.title.align = 'left'
    plt.title.text_color = TEXT_COLOR
    plt.title.text_font = TEXT_FONT
    plt.title.text_font_size = '9pt'
    plt.title.text_font_style = 'normal'

    return plt


def plot_detection_history():
    # Rectangle grid with detection history
    cols = np.shape(audio.predictions)[1]

    plt = figure(plot_width=WIDTHS[0], plot_height=HEIGHTS[1],
                 toolbar_location=None, tools="hover",
                 x_range=[-cols, 0], y_range=labels[::-1])

    plt.rect(x='x', y='y', width=0.95, height=0.8, color='color', source=HISTORY)

    # X ticks
    plt.xaxis[0].ticker = FixedTicker(ticks=np.arange(-cols, 1, 1).tolist())
    plt.xaxis[0].formatter = FuncTickFormatter(code="""
        return (tick * {} / 1000).toFixed(1) + " s"
    """.format(PREDICTION_STEP_IN_MS))
    plt.xaxis.major_tick_line_color = GRID_COLOR
    plt.xaxis.major_label_text_font_size = '7pt'
    plt.xaxis.major_label_text_font = TEXT_FONT
    plt.xaxis.major_label_text_color = TEXT_COLOR

    # X axis
    plt.xaxis.axis_line_color = None

    # Y ticks
    plt.yaxis.major_tick_line_color = None
    plt.yaxis.major_label_text_font_size = '7pt'
    plt.yaxis.major_label_text_font = TEXT_FONT
    plt.yaxis.major_label_text_color = TEXT_COLOR

    # Y axis
    plt.yaxis.axis_line_color = GRID_COLOR

    # Grid
    plt.ygrid.grid_line_color = None
    plt.xgrid.grid_line_color = None

    # Plot fill/border
    plt.background_fill_color = GRID_COLOR
    plt.outline_line_color = GRID_COLOR
    plt.min_border = 10

    # Plot title
    plt.title.text = 'Detection history:'
    plt.title.align = 'left'
    plt.title.text_color = TEXT_COLOR
    plt.title.text_font = TEXT_FONT
    plt.title.text_font_size = '9pt'
    plt.title.text_font_style = 'normal'

    # Hover tools
    hover = plt.select(dict(type=HoverTool))
    hover.tooltips = [
        ("Event", "@label"),
        ('Probability', '@pretty_value'),
    ]

    return plt


def plot_detection_last():
    # Horizontal bars with current probabilities
    plt = figure(plot_width=WIDTHS[1], plot_height=HEIGHTS[1],
                 toolbar_location=None, tools='hover',
                 x_range=[0., 1.], y_range=labels[::-1])

    plt.hbar(y='pos', height=0.9, left=0, right='value', color='color', source=DETECTION,
             name='bars', line_color=None)

    # Threshold annotation
    plt.quad(left=-0.1, right='threshold', bottom=-0.1, top=len(labels) + 1, source=THRESHOLD,
             fill_color='#000000', fill_alpha=0.1, line_color='red', line_dash='dashed')

    # X ticks
    plt.xaxis[0].ticker = FixedTicker(ticks=[0, 1])
    plt.xaxis.major_tick_line_color = GRID_COLOR
    plt.xaxis.major_label_text_font_size = '7pt'
    plt.xaxis.major_label_text_font = TEXT_FONT
    plt.xaxis.major_label_text_color = TEXT_COLOR

    # X axis
    plt.xaxis.axis_line_color = None

    # Y ticks
    plt.yaxis[0].ticker = FixedTicker(ticks=np.arange(1, len(labels) + 1, 1).tolist())
    plt.yaxis.major_label_text_font_size = '0pt'
    plt.yaxis.major_tick_line_color = None

    # Y axis
    plt.yaxis.axis_line_color = GRID_COLOR

    # Grid
    plt.xgrid.grid_line_color = None
    plt.ygrid.grid_line_color = None

    # Band fill
    plt.hbar(y=np.arange(1, len(labels) + 1, 2), height=1., left=0, right=1., color='#000000',
             alpha=0.1, level='image', name='bands')

    # Plot fill/border
    plt.outline_line_color = GRID_COLOR
    plt.min_border = 10

    # Plot title
    plt.title.text = 'Current frame:'
    plt.title.text_color = TEXT_COLOR
    plt.title.text_font = TEXT_FONT
    plt.title.text_font_size = '9pt'
    plt.title.text_font_style = 'normal'

    # Hover tools
    hover = plt.select(dict(type=HoverTool))
    hover.names = ['bars']
    hover.tooltips = [
        ('Event', '@label'),
        ('Probability', '@pretty_value'),
    ]

    return plt


def update():
    if len(audio.live_audio_feed):
        spectrogram = audio.spectrogram.copy()
        predictions = audio.predictions.copy()

        rows = len(labels)
        cols = np.shape(predictions)[1]
        pred = predictions[:, -1].tolist()

        # Push new audio
        data = dict(
            signal=[audio.live_audio_feed.pop()]
        )

        LIVE_AUDIO.data = data

        # Update spectrogram
        data = dict(
            value=[spectrogram],
        )

        SPECTROGRAM.data = data

        # Update last detection plot data
        DETECTION.data = dict(
            pos=np.arange(0, rows, 1)[::-1] + 1,  # in reversed order
            label=labels,
            value=pred,
            pretty_value=[str(to_percentage(v)) + '%' for v in pred],
            color=[colorizer(v, True, threshold=threshold.value) for v in pred]
        )

        # Update threshold line
        THRESHOLD.data = dict(
            threshold=[threshold.value - 0.01]
        )

        # Update detection history data
        data = dict(
            label=[],
            x=[],
            y=[],
            value=[],
            pretty_value=[],
            color=[],
        )

        for r in range(rows):
            for c in range(cols):
                data['label'].append(labels[r])
                data['x'].append(0.5 + c - cols)
                data['y'].append(rows - r)  # inverted order
                data['value'].append(predictions[r, c])
                data['pretty_value'].append(str(to_percentage(predictions[r, c])) + '%')
                data['color'].append(colorizer(predictions[r, c], True, threshold=threshold.value))

        HISTORY.data = data


live_audio_callback = CustomJS(args=dict(feed=LIVE_AUDIO), code="""
    if (cb_obj.attributes.name == 'mute_button') {{
        if (is_muted) {{
            is_muted = false;
            cb_obj.attributes.label = 'Mute';
            cb_obj.trigger('change');
            console.log('Playback unmuted');
        }} else {{
            is_muted = true;
            cb_obj.attributes.label = 'Unmute';
            cb_obj.trigger('change');
            console.log('Playback muted');
        }}
    }}

    var new_signal = feed.data.signal[0];

    var node = audio_context.createBufferSource();
    var buffer = audio_context.createBuffer(1, {}, {});
    var signal = buffer.getChannelData(0);

    var streaming_delay = {};

    if (next_chunk_time == 0) {{
        next_chunk_time = audio_context.currentTime + streaming_delay;
    }}


    if (!is_muted) {{
        for (var i=0; i<new_signal.length; i++) {{
            signal[i] = new_signal[i];
        }}

        node.buffer = buffer;
        node.connect(audio_context.destination);
        node.start(next_chunk_time);
    }}

    next_chunk_time = next_chunk_time + buffer.duration;
""".format(BLOCK_SIZE * PREDICTION_STEP, SAMPLING_RATE, PREDICTION_STEP_IN_MS / 1000.))

LIVE_AUDIO.js_on_change('data', live_audio_callback)

spec_plot = plot_spectrogram()
history_plot = plot_detection_history()
last_plot = plot_detection_last()
filler = Div(width=(np.sum(WIDTHS) - SLIDER_WIDTH) // 2 + 20)
filler2 = Div(width=(np.sum(WIDTHS) - SLIDER_WIDTH) // 2 + 20)

grid = column(
    row(spec_plot),
    row(history_plot, last_plot),
    row(filler, widgetbox(threshold)),
    row(filler2, Button(label='Mute', callback=live_audio_callback, name='mute_button'))
)

curdoc().title = 'EARS: Environmental Audio Recognition System'
curdoc().add_root(grid)
curdoc().add_periodic_callback(update, 100)
