import os
import collections
import itertools
from argparse import ArgumentParser

import numpy as np
import pandas

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2


def generator_from_event_file(event_file):
    """Returns a generator that yields events from an event file."""
    return event_file_loader.EventFileLoader(event_file).Load()


def generators_from_logdir(logdir):
    """Returns a list of event generators for subdirectories with event files.
    The number of generators returned should equal the number of directories
    within logdir that contain event files. If only logdir contains event files,
    returns a list of length one.
    Args:
      logdir: A log directory that contains event files.
    Returns:
      List of event generators for each subdirectory with event files.
    """
    subdirs = io_wrapper.GetLogdirSubdirectories(logdir)
    generators = {
        os.path.basename(subdir):
        itertools.chain(
            *[
                generator_from_event_file(os.path.join(subdir, f))
                for f in tf.io.gfile.listdir(subdir)
                if io_wrapper.IsTensorFlowEventsFile(os.path.join(subdir, f))
            ]
        )
        for subdir in subdirs
    }
    print(len(generators))
    return generators


def read_generator(generator, tags):
    data = sorted(
        map(
            lambda event: (event.step, event.wall_time, event.summary.value[0].simple_value, event.summary.value[0].tag),
            filter(
                lambda event: event.summary.value[0].tag in tags if event.summary.value else False,
                generator
            )        
        ),
        key = lambda event: (event[0], event[1])
    )
    table = collections.defaultdict(list)
    for event in data:
        table[event[-1]].append((event[0], event[2]))
    for tag in table:
        table[tag] = pandas.DataFrame(table[tag], columns=['step', tag]).drop_duplicates(subset=['step'], keep='last')
    return table


def write(outdir, format, intervals, tables):
    os.makedirs(outdir, exist_ok=True)
    for experiment in tables:
        if experiment == '':
            name = os.path.basename(os.path.normpath(outdir))
        else:
            name = experiment

        def priority(key):
            if key == 'iterations': return 3
            if key == 'samples': return 2
            if 'loss' in key: return 1
            return 0

        def to_format(csv, name, format, index=False):
            if format == 'csv':
                csv.to_csv(name, index=index)#, float_format='%.5f')
            elif format == 'xlsx':
                csv.to_excel(name, index=index)#, float_format='%.5f')
            else:
                raise ValueError

        for interval in intervals:
            if interval == 1:
                csv = tables[experiment]
                csv = csv[sorted(csv.columns, key=priority, reverse=True)]
                to_format(csv, f'{os.path.join(outdir, name)}.{format}', format)
            else:
                pdf = tables[experiment].iloc[interval - 1:]
                rolling = tables[experiment].rolling(interval)
                avg = rolling.mean()[interval - 1:]
                avg['samples'] = pdf['samples'].values
                avg['iterations'] = pdf['iterations'].values
                std = rolling.std()[interval - 1:]
                std['samples'] = pdf['samples'].values
                std['iterations'] = pdf['iterations'].values
                csv = avg.merge(std, on=['samples', 'iterations'], how='inner', suffixes=[' avg', ' std'], validate='one_to_one')
                csv = csv[sorted(csv.columns, key=priority, reverse=True)]
                to_format(csv, f'{os.path.join(outdir, name)}.rolling{str(interval).zfill(4)}.{format}', format)


def main(args):
    generators = generators_from_logdir(args.logdir)
    tables = {}
    for experiment in generators:
        dataframes = read_generator(generators[experiment], args.tags + ['lm loss'])
        iteration_offset = dataframes['lm loss']['step'].iloc[0] - 1
        del dataframes['lm loss']
        for tag in dataframes:
            try:
                dataframe = dataframe.merge(dataframes[tag], on='step', how='outer')
            except:
                dataframe = dataframes[tag]
        #dataframe['ppl vs samples'] = np.exp(dataframe['lm loss vs samples'])
        tables[experiment] = dataframe
        dataframe = None
        tables[experiment] = tables[experiment].rename(columns={'step': 'samples'})
        tables[experiment]['iterations'] = np.arange(len(tables[experiment]['samples'].values)) + 1 + iteration_offset
        del tables[experiment]['samples']
    write(args.outdir, args.format, args.intervals, tables)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('logdir', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'xlsx'])
    parser.add_argument('--intervals', type=int, nargs='*', default=[1])#, 100, 1000])
    parser.add_argument('--tags', nargs='*', default=['lm loss vs samples', 'grad-norm vs samples', 'num-zeros vs samples', 'params-norm vs samples'])

    args = parser.parse_args()

    samples_in_tags = [tag.endswith('vs samples') for tag in args.tags]
    assert all(samples_in_tags)

    main(args)
