import os
import shutil
import subprocess
from PlotTools import ScientificCbar
import cmocean
import scipy.linalg
from netCDF4 import Dataset
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
Attachments


# Need 'frames' directory to exist.

R2D = 180. / np.pi
D2R = np.pi / 180.

velocity_file = 'input.nc'
particle_file = 'particles.nc'
output_file = 'movie.mp4'


def merge_to_mp4(frame_filenames, movie_name, fps=12):
    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd = ['ffmpeg', '-framerate', str(fps), '-i', frame_filenames, '-y',
           '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', movie_name]
    subprocess.call(cmd, stdout=f_log, stderr=f_err)
    f_log.close()
    f_err.close()


with Dataset(velocity_file, 'r') as dset:

    lon = dset['longitude'][:]
    lat = dset['latitude'][:]

    time = dset['time'][:]

    uo = dset['uo']
    vo = dset['vo']

    with Dataset(particle_file, 'r') as pset:

        lats = pset['latitude']
        lons = pset['longitude']

        LAT_lb = np.min(lats[0, :]) * R2D
        LAT_ub = np.max(lats[0, :]) * R2D

        LON_lb = np.min(lons[0, :]) * R2D
        LON_ub = np.max(lons[0, :]) * R2D

        part_times = pset['time'][:]

        tail_time = 14 * 24 * 60 * 60.

        Ntime = len(time)
        Np_time, Nparts = lats.shape

        print(Ntime, flush=True)

        for Itime in range(Ntime):

            gridspec_props = dict(left=0.1, right=0.95,
                                  bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(
                6, 6), gridspec_kw=gridspec_props)

            KE = uo[Itime, 0, :, :]**2 + vo[Itime, 0, :, :]**2
            cv = np.max(KE)
            norm_KE = colors.LogNorm(vmin=cv/1e4, vmax=cv)

            q0 = ax.pcolormesh(lon, lat, KE, cmap='cmo.thermal', norm=norm_KE)
            plt.colorbar(q0, ax=ax)

            if (Itime > 0):
                if time[Itime] < part_times[-1]:
                    tStop = int(np.argmin(part_times < time[Itime]))
                else:
                    tStop = Np_time

                if time[0] < time[Itime] - tail_time:
                    tStart = int(np.argmin(part_times < time[Itime - 10]))
                else:
                    tStart = int(0)
                time_sub = part_times[tStart:tStop]

                s_arr = 1.0 * \
                    (time_sub - time_sub[0]) / (time_sub[-1] - time_sub[0])
                s_arr = np.tile(s_arr.reshape(len(time_sub), 1), (1, Nparts))

                ax.scatter(lons[tStart:tStop, :].ravel() * R2D, lats[tStart:tStop, :].ravel() * R2D,
                           s=s_arr.ravel(),
                           c=s_arr.ravel(),
                           cmap='cmo.algae')

            ax.set_xlim(LON_lb, LON_ub)
            ax.set_ylim(LAT_lb, LAT_ub)

            plt.savefig('frames/frame_{0:04d}.png'.format(Itime), dpi=100)
            plt.close()

            print('  ', Itime, flush=True)

merge_to_mp4('frames/frame_%04d.png', output_file)
