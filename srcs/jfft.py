# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy import signal
from numba import jit
import pandas as pd
import math
from typing import Union, List

import matplotlib.pyplot as plt
import copy
import matplotlib.dates as mdates
import datetime
import warnings

warnings.filterwarnings("ignore")

#import plt_parameters  # noqa: F401, pylint:disable=unused-import

# Jacky's FFT module
# =======================================================================
# =======================================================================
# =======================================================================


# @jit
def butter_filter(
    ts_data: Union[np.ndarray, List],
    cutoff: float,
    fs: float,
    btype: str = "lowpass",
    order=5,
):
    """
    https://zh.wikipedia.org/wiki/巴特沃斯滤波器
    Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    Butterworth Filter

    fs: 採樣頻率, sampling rate
    cutoff: 切斷頻率
    btype: {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
    """
    assert isinstance(cutoff, (float, int))
    assert isinstance(fs, (float, int))
    assert isinstance(ts_data, (np.ndarray, List))

    # @jit
    def butter_setup(cutoff: float, fs, btype, order):
        normal_cutoff = cutoff / (0.5 * fs)
        b, a = signal.butter(
            order,
            normal_cutoff,
            btype=btype,
            analog=False,
        )
        return b, a

    b, a = butter_setup(cutoff, fs, btype, order)
    # self.__yval = filter(b, a, self.__yval)
    # self.calc_fft()  # 計算頻譜 amplitude
    return signal.filtfilt(b, a, ts_data)


def find_cloest_index(val1, val2, compare_value):
    """
    找出比較接近的數值
    """
    result = 1
    if np.abs(val1 - compare_value) < np.abs(val2 - compare_value):
        # val1 比較接近
        result = 0
    return result


def find_cloest_value(val1, val2, compare_value):
    """
    找出比較接近的數值
    """
    index = find_cloest_index(val1, val2, compare_value)
    return [val1, val2][index]


class FFT_Obj:
    """
    # 以物件導向寫法，處理 fft 網格
    """

    # 建構子
    def __init__(self, yval=None, fs: int = None):
        # 定義角點
        if yval is not None:
            self.__yval: np.ndarray = yval
            self.__N: int = yval.shape[0]
            # debug, 需要改成 reshape((N)), 而非 (N, 1)
            self.__yval = self.__yval.reshape((-1,))

        if fs is not None:
            self.__fs = fs
            self.calc_fft()  # 計算頻譜 amplitude
            self.calc_fft_xf()  # 定義 xf

    ############################################################################
    ############################################################################
    ############################################################################
    def get_yval(self):
        return self.__yval

    def get_fs(self):
        return self.__fs

    def get_size(self) -> int:
        return self.__N

    def get_freq_data(self, freq_data_name: str) -> np.ndarray:
        """
        Get XF or YF data
        """
        func_pair = {
            "XF": self.get_xf,
            "YF": self.get_yf,
        }
        assert freq_data_name in func_pair.keys()
        # debug: bug fix
        # 原本 return method, add () to call the method
        return func_pair[freq_data_name]()

    def get_yf(self) -> np.ndarray:
        """
        Get YF data
        """
        return 2.0 / self.__N * np.abs(self.__yf_fft[0 : int(self.__N / 2)])

    def get_xf(self) -> np.ndarray:
        """
        Get XF data
        """
        return self.__xf

    ############################################################################
    ############################################################################
    ############################################################################
    # 計算 fft 頻譜, Amplitude
    def calc_fft(self):
        self.__yf_fft = scipy.fft.fft(self.__yval)  # 計算 fft

    # 計算 frequency 的 資訊
    def calc_fft_xf(self):
        self.__xf: np.ndarray = np.linspace(
            0.0,
            1.0 / 2.0 * self.__fs,
            int(self.__N / 2),
        )
        self.define_gap()  # 計算 xf resolution

    # 計算 xf resolution
    def define_gap(self):
        # pylint: disable=attribute-defined-outside-init
        self.gap: float = self.__fs / 2.0 / (self.__N / 2)

    # iFFT
    def calc_ifft(self, yf_fft=None):
        if yf_fft is not None:
            return np.real(scipy.ifft(yf_fft))
        return np.real(scipy.ifft(self.__yf_fft))

    # 複製 class
    def duplicate(self):
        return copy.deepcopy(self)

    def notch_filter(self, freq_val, freq_range):
        """
        # Notch Filter
        點阻濾波器
        """
        index1 = self.find_xf_index(freq_val - freq_range / 2)
        index2 = self.find_xf_index(freq_val + freq_range / 2)
        index3 = self.__N - 1 - index2
        index4 = self.__N - 1 - index1
        self.__yf_fft[index1 : index2 + 1] = 0
        self.__yf_fft[index3 : index4 + 1] = 0
        return self.calc_ifft()

    def get_freq_resolution(self):
        return self.gap

    # @jit
    def find_xf_index(self, freq_val: float) -> int:
        """
        # 確定特定頻率，對應的 xf_index
        """
        assert freq_val > 0
        assert freq_val <= np.max(self.__xf)
        delta_xf = self.get_freq_resolution()  # 頻率解析度
        index1 = int(math.ceil(freq_val / delta_xf))
        index2 = int(math.floor(freq_val / delta_xf))
        # pylint: disable=no-else-return
        if index1 > self.__xf.shape[0] - 1:
            return index2

        select_index = find_cloest_index(
            self.__xf[index1],
            self.__xf[index2],
            freq_val,
        )
        return [index1, index2][select_index]

    def find_xf_amplitude(self, freq_val: Union[int, float]) -> float:
        """
        特定頻率的 amplitude
        """

        assert isinstance(freq_val, (int, float))
        index = self.find_xf_index(freq_val)
        return self.get_yf()[index]

    def find_xf_amplitude_max(self, freq_val):
        """
        採取前後3 個數值中，最大 amplitude
        """
        index = self.find_xf_index(freq_val)

        ampl = 0
        yf = self.get_yf()
        if index == 0:
            ampl = np.max(yf[:3])
        elif index == self.get_size() - 1:
            ampl = np.max(yf[-3:])
        else:
            ampl = np.max(yf[index - 1 : index + 2])
        return ampl

    """
    # 特定頻率段的 amplitude 平均值, sum, max or min
    # type: 0, mean; 1, sum; 2, max; 3, min
    """

    def find_xf_range_amplitude(self, freq_val, freq_delta, freq_type=None):
        index1 = self.find_xf_index(freq_val - freq_delta / 2)
        index2 = self.find_xf_index(freq_val + freq_delta / 2)
        if index1 == index2:
            return self.get_yf()[index1]
        if freq_type is not None:
            # pylint: disable=no-else-return
            if freq_type == 1:
                return np.sum(self.get_yf()[index1:index2])
            elif freq_type == 2:
                return np.max(self.get_yf()[index1:index2])
            elif freq_type == 3:
                return np.min(self.get_yf()[index1:index2])
        return np.mean(self.get_yf()[index1:index2])

    def plot(
        self,
        xf_limit=None,
        amp_limit=None,
        fig_title=None,
        ax=None,
        ylabel="Amplitude",
        xlabel="Frequency",
        linewidth=None,
        linestyle=None,
        fig_transpose: bool = False,
        **kwargs,
    ):
        yf = self.get_yf()
        xf = self.get_xf()

        def ax_setup(func, content, **kwargs):
            """
            如果為 None, 則不處理
            """
            if content is not None:
                func(content, **kwargs)

        # 排除部分的 kwargs 內容
        kwargs2 = {
            key: kwargs[key] for key in kwargs.keys() if key not in ["fontsize"]
        }
        if not fig_transpose:
            # 橫軸為 xf, 縱軸為 yf

            ax.plot(
                xf,
                yf,
                linewidth=linewidth,
                linestyle=linestyle,
                **kwargs2,
            )  # 繪圖
            ax_setup(ax.set_xlabel, xlabel, **kwargs)
            ax_setup(ax.set_ylabel, ylabel, **kwargs)
            ax_setup(ax.set_xlim, xf_limit)
            ax_setup(ax.set_ylim, amp_limit)
        else:
            ax.plot(
                yf,
                xf,
                linewidth=linewidth,
                linestyle=linestyle,
                **kwargs2,
            )  # 繪圖
            ax_setup(ax.set_xlabel, ylabel, **kwargs)
            ax_setup(ax.set_ylabel, xlabel, **kwargs)
            ax_setup(ax.set_xlim, amp_limit)
            ax_setup(ax.set_ylim, xf_limit)

        ax.grid(color="grey", linestyle="--")
        ax_setup(ax.set_title, fig_title, **kwargs)

    @jit
    def export_fft_hdf(self, fname):
        yf = self.__yf_fft
        xf = self.get_xf()
        xf2 = np.append(xf, np.flip(xf, 0))
        mat = list(zip(xf2, yf, self.__yval))
        df = pd.DataFrame(mat, columns=["xf", "yf", "y"])
        df.to_hdf(
            fname,
            key="df",
            mode="w",
            complevel=4,
            complib="blosc",
        )

    # 載入輸出資料
    @jit
    def load_fft_hdf(self, fname):
        # pylint: disable=attribute-defined-outside-init
        df = pd.read_hdf(fname)
        self.__N = df.shape[0]
        self.__yf_fft = df.loc[:, "yf"]
        self.__yval = df.loc[:, "y"]
        self.__xf = df.loc[: int(self.__N / 2) - 1, "xf"]
        self.__fs = np.max(self.__xf) * 2
        self.define_gap()  # 計算 xf resolution


# =======================================================================
# =======================================================================
# =======================================================================
# pylint: disable=too-many-instance-attributes
class STFT_Obj:
    """
    # 以物件導向寫法，處理 stft 網格
    """

    # 建構子
    # @jit
    def __init__(
        self,
        yval=None,
        fs=None,
        framesz=None,
        hop=None,
        dt_list=None,
    ):
        # 定義角點

        if yval is not None:
            self.__yval = yval
            self.__N: int = yval.shape[0]

            self.__yval = self.__yval.reshape((-1,))
        if fs is not None:
            self.__fs: float = fs
            self.__framesz = framesz
            self.__hop = hop
            self.__dt_list = dt_list

            if yval is not None:
                self.T = self.__fs * self.__N
                self.calc_stft()  # 計算時頻圖

                # self.calc_fft_xf()  # 定義 xf
        assert len(self.__yval.shape) == 1

    ############################################################################
    ############################################################################
    ############################################################################
    def get_stft_param(self) -> tuple:
        return (self.__fs, self.__framesz, self.__hop)

    def get_yf(self):
        """
        取出 stft 頻譜


        get stft array
        """
        return (
            2.0
            * np.abs(
                self.__yf_stft[
                    :,
                    0 : int(self.__yf_stft.shape[1] / 2),
                ]
            )
            / int(self.__framesz * self.__fs)
        )

    def get_xf(self):
        """
        取出 stft 對應之頻率
        Get the freq list

        """
        return self.__xf

    def get_freq_resolution(self):
        """
        計算頻率域解析度
        """
        return self.__xf[1] - self.__xf[0]

    def get_timeval(self):
        """
        time value
        """
        framesamp = int(self.__framesz * self.__fs)
        hopsamp = int(self.__hop * self.__fs)
        index1 = int(framesamp / 2)
        index2 = self.__dt_list.shape[0] - 1 - int(framesamp / 2)
        myslice = slice(index1, index2 + 1, hopsamp)
        return self.__dt_list[myslice]

    ############################################################################
    ############################################################################
    ############################################################################
    # @jit, try & except 無法使用 jit
    def calc_stft(self):
        """x is the time-domain signal
        fs is the sampling frequency
        framesz is the frame size, in seconds
        hop is the the time between the start of consecutive frames, in seconds
        """

        try:
            framesamp = int(self.__framesz * self.__fs)
            hopsamp = int(self.__hop * self.__fs)
            assert framesamp < self.__yval.shape[0] / 2
        except AssertionError as e:
            message = "!!! Size of Time Series is {}\n".format(
                self.__yval.shape[0]
            )
            message += "  framesamp is {}\n".format(framesamp)
            message += "  The value of framesamp is too large!!!"
            raise TypeError(message) from e

        if abs(hopsamp - 1.0) < 0.1:
            hopsamp = 1.0
        w = np.hamming(framesamp)
        self.__yf_stft = np.array(
            [
                scipy.fft.fft(w * self.__yval[i : i + framesamp])
                for i in range(
                    0,
                    len(self.__yval) - framesamp,
                    hopsamp,
                )
            ]
        )

        self.__xf = np.linspace(
            0.0,
            1.0 / 2.0 * self.__fs,
            int(self.__yf_stft.shape[1] / 2),
        )

    @jit
    def calc_istft(self):
        """X is the short-time Fourier transform
        fs is the sampling frequency
        T is the total length of the time-domain output in seconds
        hop is the the time between the start of consecutive frames, in seconds

        """
        # x = scipy.zeros(self.__N)
        framesamp = self.__yf_stft.shape[1]
        hopsamp = int(self.__hop * self.__fs)
        count = 0
        for n, i in enumerate(range(0, self.__N - framesamp, hopsamp)):
            self.__yval[i : i + framesamp] += np.real(
                scipy.ifft(self.__yf_stft[n])
            )
            count = count + 1

    def find_xf_index(self, freq_val: float) -> int:
        """
        # 確定特定頻率，對應的 xf_index

        Find out the index of specific frequency
        """
        delta_xf = self.get_freq_resolution()  #
        index1 = int(math.ceil(freq_val / delta_xf))
        index2 = int(math.floor(freq_val / delta_xf))

        index = 0
        # pylint: disable=unsubscriptable-object
        # pylint: disable=no-else-return
        if index1 > self.__xf.shape[0] - 1:
            return index2
        elif index2 < 0:
            return index1

        if np.abs(self.__xf[index1] - freq_val) < np.abs(
            self.__xf[index2] - freq_val
        ):
            index = index1
        else:
            index = index2
        return index

    # @jit
    def plot(
        self,
        xf_limit=None,
        amp_limit=None,
        fig_title=None,
        ylabel=None,
        ax=None,
        gridcolor="slategrey",
        **kwargs,
    ):
        yf_stft = self.get_yf().transpose()
        xf = self.get_xf()

        if xf_limit is not None:
            xf_index = [
                self.find_xf_index(xf_limit[0]),
                self.find_xf_index(xf_limit[1]),
            ]
            xf = xf[xf_index[0] : xf_index[1]]
            yf_stft = yf_stft[xf_index[0] : xf_index[1], :]

        tval = self.get_timeval()
        x_lims = [tval[0], tval[-1]]

        # You can then convert these datetime.datetime objects to the correct
        # format for matplotlib to work with.
        x_lims = mdates.date2num(x_lims)
        extent = [
            x_lims[0],
            x_lims[1],
            xf[0]
            - self.get_freq_resolution() / 2,  # 頻率的上下緣, xf[0] 為 Block Center
            xf[-1] + self.get_freq_resolution() / 2,
        ]

        if amp_limit is None:
            amp_limit = [None, None]
        im = ax.imshow(
            yf_stft,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            extent=extent,
            cmap="jet",
            vmin=amp_limit[0],
            vmax=amp_limit[1],
        )
        ax.xaxis_date()
        # This simply sets the x-axis data to diagonal so it fits better.
        ax.grid(color=gridcolor, alpha=0.7, linestyle="--")
        ax.set_xlabel("Time", **kwargs)
        plt.colorbar(im, ax=ax)
        if ylabel is not None:
            ax.set_ylabel(ylabel, **kwargs)
        if fig_title is not None:
            ax.set_title(fig_title, **kwargs)


# Inverse short time fourier transform
@jit
def istft(X, fs, T, hop):
    """X is the short-time Fourier transform
    fs is the sampling frequency
    T is the total length of the time-domain output in seconds
    hop is the the time between the start of consecutive frames, in seconds

    """
    x = np.zeros(T * fs)
    framesamp = X.shape[1]
    hopsamp = int(hop * fs)
    count = 0
    for n, i in enumerate(range(0, x.shape[0] - framesamp, hopsamp)):
        x[i : i + framesamp] += np.real(scipy.ifft(X[n]))
        count = count + 1
    return x


# Short time fourier transform
def stft(x, fs, framesz, hop):
    """x is the time-domain signal
    fs is the sampling frequency
    framesz is the frame size, in seconds
    hop is the the time between the start of consecutive frames, in seconds

    """
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    if abs(hopsamp - 1.0) < 0.1:
        hopsamp = 1.0
    w = np.hamming(framesamp)
    framesamp = int(framesz * fs)
    X = np.array(
        [
            scipy.fft.fft(w * x[i : i + framesamp])
            for i in range(0, len(x) - framesamp, hopsamp)
        ]
    )
    return X


@jit
def stft_xf(yf, fs):
    xf = np.linspace(0.0, 1.0 / 2.0 * fs, yf.shape[1] / 2)
    return xf


@jit
def stft_timeval(xval, yf, fs, framesz):
    framesamp = int(framesz * fs)
    timeval = np.linspace(
        xval[int(framesamp / 2)],
        xval[xval.shape[0] - 1 - int(framesamp / 2)],
        yf.shape[0],
    )
    timeval_dt = []
    # pylint: disable=unsubscriptable-object
    for i in range(0, timeval.shape[0]):
        timeval_dt.append(datetime.datetime.fromtimestamp(timeval[i]))
    timeval_dt2 = np.array(timeval_dt)
    return timeval, timeval_dt2
