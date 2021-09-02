import pathlib
import pickle
import sys
import datetime as dt
import logging

from IPython import embed

from cwd import cwd


def main():
    if not sys.argv[1:]:
        raise SystemExit('No argument given.')
    try:
        limit = float(sys.argv[1])
    except:
        raise SystemExit(f'Got unparsable float {sys.argv[1]!r}')

    # Locate output data
    timestamp = lambda: dt.datetime.now().isoformat()[:19].replace(':', '')
    output = cwd / pathlib.Path('__output__') / timestamp()
    output.mkdir(exist_ok=True)

    # Prepare logging
    logging.basicConfig(
        filename=output / f'lim-{limit:3.1f}.log',
        # encoding='utf-8',  # Py3.9+
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w',
        level=logging.DEBUG,
    )
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # Load pre-constructed datasets and state
    log.info('Load pre-imposed hi-res grid')
    fname_grid_hi = cwd / 'grid_hi.pkl'
    assert fname_grid_hi.is_file()
    with open(fname_grid_hi, 'rb') as fsock:
        grid_hi = pickle.load(fsock)
    log.info('Hi-res grid loaded')

    log.info('Load pre-imposed hi-res grid')
    fname_grid_lo = cwd / 'grid_lo.pkl'
    assert fname_grid_lo.is_file()
    with open(fname_grid_lo, 'rb') as fsock:
        grid_lo = pickle.load(fsock)
    log.info('Lo-res grid loaded')

    # embed()

    log.info(f'*** LIMIT: {limit:3.1f} ***')
    
    # Specify criteria for overwriting the values in the smaller cells with thatr of a bigger, containing cell

    # From the experimental cummulative distribution function of the IQR values,
    # more than 90 % of the cells have an IQR lower than 2 [mm / yr].
    filters = [
        lambda row: row[('VEL_V', 'iqr')] > limit,
    ]    
    # Shortcut: specify only the labels of the aggregate methods specified above
    stat_columns_wanted = ('mean', 'std', 'median', 'iqr')
    columns_imposable = grid_hi.get_columns_imposable(stat_columns_wanted)

    # embed()

    # Using the filter, and the lower-resolution grid,
    # overwrite the cells in the higher-resolution grid.
    log.info('Impose start')
    # log.info(columns_imposable)
    # log.debug(limit)
    grid_hi.impose(grid_lo, filters=filters, columns=columns_imposable)
    log.info('Impose end')

    # Save the imposed grid
    log.info('Saving imposed grid to separate pickle.')
    size_x, size_y = grid_hi.info.n_points_x, grid_hi.info.n_points_y
    with open(output / f'insar_grid_{size_x}x{size_y}_iqr-lim-{limit:3.1f}.pkl', 'wb+') as fsock:
        pickle.dump(grid_hi, fsock)
    log.info('Saved')
    
    # Save data from the high-resolution grid
    size_x, size_y = grid_hi.info.n_points_x, grid_hi.info.n_points_y
    for column in grid_hi.columns_imposed:
        output_name = column if isinstance(column, str) else '_'.join(column)
        ofname = output / f'insar_grid_{size_x}x{size_y}_iqr-lim-{limit:3.1f}_{output_name}.tif'
        log.info(f'Writing data to {ofname}')
        grid_hi.save(ofname, column)
        log.info('Saved')

    # Also save the original data
    for column in grid_hi.data_columns:
        output_name = column if isinstance(column, str) else '_'.join(column)
        ofname = output / f'insar_grid_{size_x}x{size_y}_iqr-lim-{limit:3.1f}_{output_name}.tif'
        log.info(f'Writing data to {ofname}')
        grid_hi.save(ofname, column)
        log.info('Saved')

    # Save data from the low-resolution grid
    size_x, size_y = grid_lo.info.n_points_x, grid_lo.info.n_points_y
    for column in grid_lo.data_columns:
        output_name = column if isinstance(column, str) else '_'.join(column)
        ofname = output / f'insar_grid_{size_x}x{size_y}_iqr-lim-{limit:3.1f}_{output_name}.tif'
        log.info(f'Writing data to {ofname}')
        grid_lo.save(ofname, column)
        log.info('Saved')


if __name__ == '__main__':
    main()

