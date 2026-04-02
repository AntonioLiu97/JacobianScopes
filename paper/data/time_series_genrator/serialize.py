from functools import partial
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

colors = {1: 'lightseagreen', 0.1: 'dodgerblue', 0.01: 'blue'}

def closest_color(width, colors):
    return colors[min(colors.keys(), key=lambda k: abs(k-width))]

class MultiResolutionPDF:
    """
    A class for managing and visualizing probability density functions (PDFs)
    in a multi-resolution format.

    This class allows for adding data in the form of bins, normalizing the bins, 
    computing statistical properties (mean, mode, and standard deviation), plotting 
    the PDF, and evaluating the PDF at a given point.

    Attributes:
        bin_center_arr (numpy.array): Stores the centers of the bins.
        bin_width_arr (numpy.array): Stores the widths of the bins.
        bin_height_arr (numpy.array): Stores the heights of the bins.
        mode (float): The mode of the PDF, computed in `compute_stats`.
        mean (float): The mean of the PDF, computed in `compute_stats`.
        sigma (float): The standard deviation of the PDF, computed in `compute_stats`.
    """
    def __init__(self, prec = None):
        """
        Constructor for the MultiResolutionPDF class.

        Initializes arrays for bin centers, widths, and heights. Statistical properties
        (mode, mean, sigma) are initialized to None.
        """
        if prec == None:
            self.bin_center_arr = np.array([])
            self.bin_width_arr = np.array([])
            self.bin_height_arr = np.array([])
        else:
            self.bin_width_arr = np.array([0.1**(prec-1)] * 10**prec)
            self.bin_center_arr = np.linspace(0,10,10**prec,endpoint=False) + self.bin_width_arr/2
            self.bin_height_arr = np.array([0.1] * 10**prec)
            # self.check_gap_n_overlap()
            # self.normalize(report=True)
        self.mode = None
        self.mean = None
        self.sigma = None

    def add_bin(self, center_arr, width_arr, height_arr, idx = None):
        """
        Adds bins to the PDF.
        Do not normalize because PDF may need multiple add_bin operations

        Args:
            center_arr (array_like): Array or list of bin centers.
            width_arr (array_like): Array or list of bin widths.
            height_arr (array_like): Array or list of bin heights.

        Raises:
            AssertionError: If the lengths of center_arr, width_arr, and height_arr are not equal.
        """
        # print(len(center_arr))
        # print(len(width_arr))
        # print(len(height_arr))
        assert len(center_arr) == len(width_arr) == len(height_arr), "center_arr, width_arr, height_arr must have the same length"
        if idx is None: # insert to index position
            self.bin_center_arr = np.append(self.bin_center_arr, center_arr)
            self.bin_width_arr = np.append(self.bin_width_arr, width_arr)
            self.bin_height_arr = np.append(self.bin_height_arr, height_arr)
        else: # append
            self.bin_center_arr = np.insert(self.bin_center_arr, idx, center_arr)
            self.bin_width_arr = np.insert(self.bin_width_arr, idx, width_arr)
            self.bin_height_arr = np.insert(self.bin_height_arr, idx, height_arr) 
        # self.normalize()                       
            
    def sort_by_center(self):
        """
        Sorts the bins by their centers.
        """
        if not np.all(np.diff(self.bin_center_arr) >= 0): # check if bin_center_arr is already sorted
            sort_indices = np.argsort(self.bin_center_arr) # sort by bin_center_arr
            self.bin_center_arr = self.bin_center_arr[sort_indices]
            self.bin_width_arr = self.bin_width_arr[sort_indices]
            self.bin_height_arr = self.bin_height_arr[sort_indices]
    
    def delete_by_idx(self, idx):
        """
        Deletes bins from the PDF by their indices.

        Args:
            idx (int or array_like): Index or list of indices of the bins to delete.
        """
        self.bin_center_arr = np.delete(self.bin_center_arr, idx)
        self.bin_width_arr = np.delete(self.bin_width_arr, idx)
        self.bin_height_arr = np.delete(self.bin_height_arr, idx) 
        
    def refine(self, Multi_PDF):
        """
        Refines the PDF by merging it with another MultiResolutionPDF.
        Reduce to add_bin if self empty

        Args:
            Multi_PDF (MultiResolutionPDF): Another MultiResolutionPDF to merge with.
        """
        if len(self.bin_center_arr) == 0:
            self.add_bin(Multi_PDF.bin_center_arr, Multi_PDF.bin_width_arr, Multi_PDF.bin_height_arr)
        else:    
            Multi_PDF.normalize()
            assert isinstance(Multi_PDF, MultiResolutionPDF), "Input must be an instance of MultiResolutionPDF"
            
            self.sort_by_center()
            right_edges = self.bin_center_arr + self.bin_width_arr/2
            insert_index = np.searchsorted(right_edges, Multi_PDF.bin_center_arr.min())
            insert_index_right = np.searchsorted(right_edges, Multi_PDF.bin_center_arr.max())
            # print(right_edges)
            # print(Multi_PDF.bin_center_arr)
            assert insert_index == insert_index_right, "refinement cannot straddle coarse bins"
            prefactor = self.bin_width_arr[insert_index] * self.bin_height_arr[insert_index] # probability of coase bin to replace
                    
            Multi_PDF.bin_height_arr *= prefactor
            self.delete_by_idx(insert_index)
            ### add bin, but to specific index
            self.add_bin(Multi_PDF.bin_center_arr, Multi_PDF.bin_width_arr, Multi_PDF.bin_height_arr, insert_index)

            # print(self.bin_center_arr)
            assert np.all(np.diff(self.bin_center_arr) >= 0), "final array should be sorted"
            self.check_gap_n_overlap()
            self.normalize()
            
    def coarsen(self, coarse_bin_centers, coarse_bin_widths):
        """
        Replace fine bins using coarse ones. This is for plotting purposes only.

        Args:
            coarse_bin_centers (np.ndarray): The centers of the coarse bins.
            coarse_bin_widths (np.ndarray): The widths of the coarse bins.
        """
        for coarse_bin_center, coarse_bin_width in zip(coarse_bin_centers, coarse_bin_widths):
            # Find the indices of the fine bins that fall within the coarse bin
            indices = np.where((self.bin_center_arr >= coarse_bin_center - coarse_bin_width / 2) &
                            (self.bin_center_arr <= coarse_bin_center + coarse_bin_width / 2))[0]

            if len(indices) == 0:
                continue

            # Compute the total height of the fine bins
            total_height = np.sum(self.bin_height_arr[indices] * self.bin_width_arr[indices])

            # Replace the fine bins with the coarse bin
            self.bin_center_arr = np.delete(self.bin_center_arr, indices)
            self.bin_width_arr = np.delete(self.bin_width_arr, indices)
            self.bin_height_arr = np.delete(self.bin_height_arr, indices)

            self.bin_center_arr = np.append(self.bin_center_arr, coarse_bin_center)
            self.bin_width_arr = np.append(self.bin_width_arr, coarse_bin_width)
            self.bin_height_arr = np.append(self.bin_height_arr, total_height / coarse_bin_width)

        # Sort the bins by their centers
        sort_indices = np.argsort(self.bin_center_arr)
        self.bin_center_arr = self.bin_center_arr[sort_indices]
        self.bin_width_arr = self.bin_width_arr[sort_indices]
        self.bin_height_arr = self.bin_height_arr[sort_indices]
        
    
    
    def load_from_num_prob(self, num_slice, prob_slice):
        """
        Loads the PDF from a given number slice and probability slice.

        Args:
            num_slice (array_like): The number slice to load from.
            prob_slice (array_like): The probability slice to load from.
        """
        assert len(num_slice) == len(prob_slice), "number of digits must equal number of probs"
        preceding_digits = None
        for idx, probs in enumerate(prob_slice):
            single_digit_PDF = MultiResolutionPDF()
            single_digit_PDF.load_from_prec_digits_prob(preceding_digits, probs)
            self.refine(single_digit_PDF)
            preceding_digits = num_slice[:idx+1]
            
    def load_from_prec_digits_prob(self, preceding_digits, probs):
        """
        Loads the PDF from a given preceding digits and probabilities of the last digit.

        Args:
            preceding_digits (array_like): The preceding digits, 
                which imply left_edge and bin_width
            probs (array_like): Distribution of next digit
        """
        assert len(probs.shape) == 1, "probs must be 1D"
        if preceding_digits is None:
            prec_len = 0
            w = 1
            left_edge = 0
        else:
            prec_len = len(preceding_digits)
            w = 0.1**prec_len
            left_edge = int(preceding_digits) * 10 * w
        x_coords = np.linspace(left_edge,left_edge+10 * w, 10, endpoint = False) + 0.5 * w
        
        self.add_bin(center_arr = x_coords,
                    width_arr = np.ones(10) * w,
                    height_arr = probs)
        self.normalize()    
    
    def normalize(self, report = False):
        """
        Normalizes the PDF so that the total area under the bins equals 1.
        Prints the total area before and after normalization.
        """
        total_area = np.sum(self.bin_width_arr * self.bin_height_arr)
        if total_area == 1.:
            if report:
                print('already normalized')
        else:
            if report:
                print('total area before normalization:', total_area)
            self.bin_height_arr = self.bin_height_arr / total_area
            
    def compute_stats(self):  
        """
        Computes and updates the statistical properties of the PDF: mean, mode, and standard deviation (sigma).
        """
        self.mean = np.sum(self.bin_center_arr * self.bin_width_arr * self.bin_height_arr)
        self.mode = self.bin_center_arr[np.argmax(self.bin_height_arr)]
        variance = np.sum(
            (self.bin_center_arr-self.mean) ** 2 * self.bin_height_arr * self.bin_width_arr
        )
        self.sigma = np.sqrt(variance)
            
    def compute_moment(self, n):
        """
        Computes the nth mean-centered moment of the PDF.

        Args:
            n (int): The order of the moment to compute.

        Returns:
            float: The nth moment of the PDF.
        """
        if self.mean is None:
            self.compute_stats()
        return np.sum(
            (self.bin_center_arr - self.mean) ** n * self.bin_height_arr * self.bin_width_arr
            )
        
    def rescale_temperature(self, alpha):
        """
        Rescale bins as if the original temperature 
        of softmax is scaled from T to alpha T
        """
        self.bin_height_arr = self.bin_height_arr ** (1/alpha)
        self.normalize()
        
    def check_gap_n_overlap(self):
        assert np.allclose(self.bin_center_arr[1:] - self.bin_width_arr[1:]/2, 
                           self.bin_center_arr[:-1] + self.bin_width_arr[:-1]/2), "bin overlap detected"
        
    def discretize(self, func, mode = 'pdf'):
        """
        Args:
            func: a function supported on self.bin_center_arr.
                  should be implmented using numpy operations for parallelization
            mode: 'pdf': approximate probability of bin using its center
                  'cdf': integrate over bin 
        Populate bin height by dicretizng
        """
        if mode == 'pdf':
            self.bin_height_arr = func(self.bin_center_arr)
        elif mode == 'cdf':
            right_edge = self.bin_center_arr + self.bin_width_arr/2
            left_edge = self.bin_center_arr - self.bin_width_arr/2
            prob_arr = func(right_edge) - func(left_edge)
            self.bin_height_arr = prob_arr / self.bin_width_arr
        self.normalize()
        
    def BT_dist(self, Multi_PDF):
        """
        Calculate the Bhattacharyya distance with another Multi_PDF object
        """          
        assert np.allclose(self.bin_center_arr, Multi_PDF.bin_center_arr), "Only PDFs of the same discretization are comparable"
        weighted_PQ_arr = np.sqrt(self.bin_height_arr * Multi_PDF.bin_height_arr) * self.bin_width_arr
        return np.sqrt(-np.log(np.sum(weighted_PQ_arr)))
    
    def Hel_dist(self, Multi_PDF):
        """
        Calculate the Hellinger distance with another Multi_PDF object
        """          
        assert np.allclose(self.bin_center_arr, Multi_PDF.bin_center_arr), f"Only PDFs of the same discretization are comparable"
        weighted_PQ_arr = np.sqrt(self.bin_height_arr * Multi_PDF.bin_height_arr) * self.bin_width_arr
        sum_weighted_PQ = np.sum(weighted_PQ_arr)
        clamped_value = np.clip(2 - 2 * sum_weighted_PQ, 0, None)
        sqrt_value = np.sqrt(clamped_value)
        
        if np.isnan(sqrt_value) or np.isinf(sqrt_value):
            print(f"Invalid sqrt value: {sqrt_value}, clamped_value: {clamped_value}, sum_weighted_PQ: {sum_weighted_PQ}")
        
        return sqrt_value
    
    def KL_div(self, Multi_PDF):
        """
        Calculate the KL divergence D_KL(self||Multi_PDF)
        Prone to numerical instabilities
        """          
        assert np.allclose(self.bin_center_arr, Multi_PDF.bin_center_arr), "Only PDFs of the same discretization are comparable"
        # log_ratio = np.log(self.bin_height_arr) - np.log(Multi_PDF.bin_height_arr)
        log_ratio = np.log(self.bin_height_arr + 1e-10) - np.log(Multi_PDF.bin_height_arr + 1e-10) 
        weighted_log_ratio = log_ratio * self.bin_height_arr * self.bin_width_arr
        return np.sum(weighted_log_ratio)
    
    def sKL(self, Multi_PDF):
        """
        Calculate the symmetrized KL divergence 
        Prone to numerical instabilities
        """          
        assert np.allclose(self.bin_center_arr, Multi_PDF.bin_center_arr), "Only PDFs of the same discretization are comparable"
        log_ratio = np.log(self.bin_height_arr + 1e-10) - np.log(Multi_PDF.bin_height_arr + 1e-10)       
        weighted_log_ratio = log_ratio * (self.bin_height_arr * self.bin_width_arr - Multi_PDF.bin_height_arr * Multi_PDF.bin_width_arr)
        return np.sqrt(np.sum(weighted_log_ratio))    
    
    def L2_dist(self,Multi_PDF):
        assert np.allclose(self.bin_center_arr, Multi_PDF.bin_center_arr), "Only PDFs of the same discretization are comparable"
        L2_dist = np.sum((self.bin_height_arr - Multi_PDF.bin_height_arr) ** 2 * self.bin_width_arr)
        return L2_dist
    
    def plot(self, ax=None, log_scale=False, statistic = True, uniform_color = None, alpha=1):
        """
        Plots the PDF as a bar chart.

        Args:
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object. If None, a new figure and axis are created.
            log_scale (bool, optional): If True, sets the y-axis to logarithmic scale.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 4), dpi=100)

        # ax.bar(self.bin_center_arr, self.bin_height_arr, width=self.bin_width_arr, align='center', color='black', alpha=0.5)
        # Define colors for different bin widths

        
        # Iterate over bins and plot with corresponding color
        for center, width, height in zip(self.bin_center_arr, self.bin_width_arr, self.bin_height_arr):
            if uniform_color is not None:
                color = uniform_color
            else:
                color = closest_color(width, colors)
            ax.bar(center, height, width=width, align='center', color=color, alpha=alpha)
        

        if log_scale:
            ax.set_yscale('log')

        if statistic:
            ax.vlines(self.mean, 0, np.max(self.bin_height_arr), color='blue', label='Mean', lw=2)
            ax.vlines(self.mode, 0, np.max(self.bin_height_arr), color='lightblue', label='Mode', lw=2)
            # Visualize sigma as horizontal lines
            ax.hlines(y=np.max(self.bin_height_arr), xmin=self.mean - self.sigma, xmax=self.mean + self.sigma, color='g', label='Sigma', lw=2)
            ax.legend()

        # If ax was None, show the plot
        if ax is None:
            plt.show()

    def value_at(self, x):
        for center, width, height in zip(self.bin_center_arr, self.bin_width_arr, self.bin_height_arr):
            if center - width / 2 <= x <= center + width / 2:
                return height
        return 0
    
def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.
    
    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    bs = val.shape[0]
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base**(max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base**(max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base**(-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base**(-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits

def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """
    Convert a string representation in a specified base back to numbers.

    Parameters:
    - sign (np.array): The sign of the numbers.
    - digits (np.array): Digits of the numbers in the specified base.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - half_bin_correction (bool): If True, adds 0.5 of the smallest bin size to the number.

    Returns:
    - np.array: Numbers corresponding to the given base representation.
    """
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped/base**powers, axis=-1)

    if half_bin_correction:
        val += 0.5/base**prec

    return sign * val

@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.

    Attributes:
    - base (int): The base for number representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - signed (bool): If True, allows negative numbers. Default is False.
    - fixed_length (bool): If True, ensures fixed length of serialized string. Default is False.
    - max_val (float): Maximum absolute value of number for serialization.
    - time_sep (str): Separator for different time steps.
    - bit_sep (str): Separator for individual digits.
    - plus_sign (str): String representation for positive sign.
    - minus_sign (str): String representation for negative sign.
    - half_bin_correction (bool): If True, applies half bin correction during deserialization. Default is True.
    - decimal_point (str): String representation for the decimal point.
    """
    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ' ,'
    bit_sep: str = ' '
    plus_sign: str = ''
    minus_sign: str = ' -'
    half_bin_correction: bool = True
    decimal_point: str = ''
    missing_str: str = ' Nan'

def serialize_arr(arr, settings: SerializerSettings):
    """
    Serialize an array of numbers (a time series) into a string based on the provided settings.

    Parameters:
    - arr (np.array): Array of numbers to serialize.
    - settings (SerializerSettings): Settings for serialization.

    Returns:
    - str: String representation of the array.
    """
    # max_val is only for fixing the number of bits in nunm2repr so it can be vmapped
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"
    
    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), f"unsigned arr must be >= 0"
        plus_sign = minus_sign = ''
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign
    
    vnum2repr = partial(vec_num2repr,base=settings.base,prec=settings.prec,max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr),np.zeros_like(arr),arr))
    ismissing = np.isnan(arr)
    
    def tokenize(arr):
        return ''.join([settings.bit_sep+str(b) for b in arr])
    
    bit_strs = []
    for sign, digits,missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate([digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]])
        digits = tokenize(digits)
        sign_sep = plus_sign if sign==1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += settings.time_sep # otherwise there is ambiguity in number of digits in the last time step
    return bit_str

def deserialize_str(bit_str, settings: SerializerSettings, ignore_last=False, steps=None):
    """
    Deserialize a string into an array of numbers (a time series) based on the provided settings.

    Parameters:
    - bit_str (str): String representation of an array of numbers.
    - settings (SerializerSettings): Settings for deserialization.
    - ignore_last (bool): If True, ignores the last time step in the string (which may be incomplete due to token limit etc.). Default is False.
    - steps (int, optional): Number of steps or entries to deserialize.

    Returns:
    - None if deserialization failed for the very first number, otherwise 
    - np.array: Array of numbers corresponding to the string.
    """
    # ignore_last is for ignoring the last time step in the prediction, which is often a partially generated due to token limit
    orig_bitstring = bit_str
    bit_strs = bit_str.split(settings.time_sep)
    # remove empty strings
    bit_strs = [a for a in bit_strs if len(a) > 0]
    if ignore_last:
        bit_strs = bit_strs[:-1]
    if steps is not None:
        bit_strs = bit_strs[:steps]
    vrepr2num = partial(vec_repr2num,base=settings.base,prec=settings.prec,half_bin_correction=settings.half_bin_correction)
    max_bit_pos = int(np.ceil(np.log(settings.max_val)/np.log(settings.base)).item())
    sign_arr = []
    digits_arr = []
    try:
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(settings.minus_sign):
                sign = -1
            elif bit_str.startswith(settings.plus_sign):
                sign = 1
            else:
                assert settings.signed == False, f"signed bit_str must start with {settings.minus_sign} or {settings.plus_sign}"
            bit_str = bit_str[len(settings.plus_sign):] if sign==1 else bit_str[len(settings.minus_sign):]
            if settings.bit_sep=='':
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(settings.bit_sep)]
            if settings.fixed_length:
                assert len(bits) == max_bit_pos+settings.prec, f"fixed length bit_str must have {max_bit_pos+settings.prec} bits, but has {len(bits)}: '{bit_str}'"
            digits = []
            for b in bits:
                if b==settings.decimal_point:
                    continue
                # check if is a digit
                if b.isdigit():
                    digits.append(int(b))
                else:
                    break
            #digits = [int(b) for b in bits]
            sign_arr.append(sign)
            digits_arr.append(digits)
    except Exception as e:
        print(f"Error deserializing {settings.time_sep.join(bit_strs[i-2:i+5])}{settings.time_sep}\n\t{e}")
        print(f'Got {orig_bitstring}')
        print(f"Bitstr {bit_str}, separator {settings.bit_sep}")
        # At this point, we have already deserialized some of the bit_strs, so we return those below
    if digits_arr:
        # add leading zeros to get to equal lengths
        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0]*(max_len-len(digits_arr[i])) + digits_arr[i]
        return vrepr2num(np.array(sign_arr), np.array(digits_arr))
    else:
        # errored at first step
        return None
