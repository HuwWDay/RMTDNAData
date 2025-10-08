import cmath
from unicodedata import name
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Code for generating thinned Wigner's 

# --------------------------
# Step 1: Haar unitary + COE
# --------------------------

def haar_unitary(n):
    """Sample an n x n Haar unitary using QR of a Ginibre matrix."""
    X = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(X)
    d = np.diag(R)
    phases = d / np.abs(d)
    phases[np.abs(d) == 0] = 1.0
    Q = Q * phases.conj()
    return Q

def generate_eigenangles_coe(n, samples):
    """
    Generate eigenangles (0, 2π) for 'samples' COE matrices of size n.
    Returns a 2D array of shape (n, samples), where each column is one matrix's eigenangles.
    """
    angles_all = np.zeros((n, samples))
    for s in range(samples):
        W = haar_unitary(n)
        U = W.T @ W
        eigs = np.linalg.eigvals(U)
        angles = np.mod(np.angle(eigs), 2*np.pi)
        angles_all[:, s] = angles

        # Progress indicator
        if (s + 1) % 100 == 0:
            print(f"Generated {(s + 1)/samples*100:.1f}% of COE samples", end='\r')

    return angles_all


# --------------------------
# Step 2: Thinning + spacings
# --------------------------

def thin_and_spacings_from_array(angles_array, p):
    """
    Compute thinned circular spacings for a batch of COE matrices.
    
    Parameters
    ----------
    angles_array : ndarray of shape (n, samples)
        Each column is eigenangles of a single COE matrix.
    p : float
        Probability to keep each eigenangle.
    
    Returns
    -------
    all_spacings : 1D ndarray
        Flattened array of normalized spacings from all matrices.
    """
    n, samples = angles_array.shape
    all_spacings = []

    for s in range(samples):
        angles = angles_array[:, s]
        keep = np.random.rand(n) < p
        kept = angles[keep]
        if kept.size < 2:
            continue  # skip matrices with fewer than 2 kept angles

        kept.sort()
        # circular spacing
        gaps = np.diff(np.concatenate([kept, [kept[0] + 2*np.pi]]))
        gaps /= np.mean(gaps)  # normalize to mean spacing 1
        all_spacings.extend(gaps)

    return np.array(all_spacings)


# File extracting, processing and matching

def file_extractor(name, spacingtype = "Interorigin_spacing") -> pd.DataFrame:
    """
    Extracts the data from the file and normalises the Interorigin spacing.
    :param name: Name of the file to extract data from.
    :param spacingtype: Type of spacing to normalise. Default is "Interorigin_spacing".
    :return: A pandas DataFrame containing the normalised Interorigin spacing.
    """
    dna = pd.read_csv(f'processeddata/' + name + "_" + spacingtype + ".csv", sep='\t')
    # Remove the first row if its NaN
    if dna.iloc[0].isnull().all():
        dna = dna.iloc[1:]

    MeanSpacing = dna["Interorigin_spacing"].mean()
    assert MeanSpacing != 0, "Mean spacing is not 0"
    dna["normalised"] = dna["Interorigin_spacing"]/MeanSpacing
    #print("Mean spacing: ", MeanSpacing)
    #print("New mean spacing: ", dna["normalised"].mean())

    return dna

def dna_histogram(dna: pd.DataFrame, Lower = 0, Upper = 3, binno = 50) -> tuple:
    """
    Creates a histogram of the normalised Interorigin spacing.
    :param dna: A pandas DataFrame containing the normalised Interorigin spacing.
    :param Lower: Lower bound for the histogram. Default is 0.
    :param Upper: Upper bound for the histogram. Default is 3.
    :param binno: Number of bins for the histogram. Default is 50.
    :return: A tuple containing the histogram and the bin edges.
    """
    # Create a histogram of the normalised Interorigin spacing
    #assert np.mean(dna['normalised']) == 1, "Mean of normalised Interorigin spacing is not 1"
    print(np.mean(dna['normalised']))


    # Normalise the histogram to sum to 1
    dnahist, bins = np.histogram(dna['normalised'], bins=binno, range=(Lower, Upper), density=True)
    # Make sure the mean of the histogram is 1
    #assert np.mean(dnahist) == 1, f"Mean of the histogram is not 1, it is {np.mean(dnahist)}"
    
    # Make area under the histogram equal to 1
    bin_width = np.diff(bins)[0]
    area = np.sum(dnahist * bin_width)
    dnahist = dnahist / area
    newarea = np.sum(dnahist * bin_width)
    assert np.isclose(newarea, 1), "Area under the histogram is not 1"
    #print(bins)
    #print(dnahist)
    #print(len(dnahist), len(bins))
    return dnahist, bins

def dna_fit_p(dnahist: np.ndarray, ThinnedDict, p = 1):
    # Basically want to load in the eigenvalues for a given p value
    # test to see how well the data fits the distribution
    # and then keep doing that until we find the best fit
    p = round(p, 2)  # Round p to 2 decimal places
    print("p=", p)
    Thinned = ThinnedDict[p]
    assert len(Thinned) == len(dnahist), f"Length of Thinned and dnahist do not match. Length of Thinned: {len(Thinned)}, Length of dnahist: {len(dnahist)}"
    # Make Thinnedcdf
    ThinnedCDF = np.cumsum(Thinned * np.diff(ThinnedDict["Bins"])[0])
    # Make dnahistcdf
    dnahistCDF = np.cumsum(dnahist * np.diff(ThinnedDict["Bins"])[0])
    # Calculate RMSE between the two CDFs
    cdf_RMSE = np.sqrt(np.sum((dnahistCDF - ThinnedCDF)**2)/len(dnahistCDF))
    #print(f"RMSE for p = {p}: ", RMSE)
    return cdf_RMSE

def min_p_searcher(dnahist: np.ndarray, ThinnedDict, name):
    """
    Load in dna data and search for the minimum p value that fits the data.
    :param dnahist: A numpy array containing the histogram of the normalised Interorigin spacing.
    :param EigenDis: Directory containing the eigenvalues for different p values.
    :return: The minimum p value that fits the data.
    """

    # Coarse search
    p_list = np.arange(0.01, 1.01, 0.01)
    error_list = []
    for p in p_list:

        rounded_p = round(float(p), 2)  # Ensure p matches dictionary key
        RMSE = dna_fit_p(dnahist, ThinnedDict, rounded_p)
        error_list.append(RMSE)
    
    # Plot the errors for search
    plt.figure(figsize=(10, 6))
    plt.plot(p_list, error_list, marker='o', linestyle='-', color='b')
    plt.title(f'CDF RMSE vs p value for {name}', fontsize=14)
    plt.xlabel('p value', fontsize=12)
    plt.ylabel('CDF RMSE', fontsize=12)
    plt.grid(True)
    plt.savefig(f'DNAPlots/ThinningError_{name}.png', dpi=300)
    plt.show()

    # Find the minimum error and corresponding p value
    min_error = min(error_list)
    min_p = p_list[np.argmin(error_list)]
    if min_p == 0.01 or min_p == 0.02:
        # These guys are janky, so reassign to 0.03
        min_p = 0.03
        min_error = error_list[2]
        print("Min p was 0.01 or 0.02, reassigning to 0.03")
    print(f"Minimum p value in coarse search: {min_p} with CDF RMSE: {min_error}")

    return min_p, min_error

def plot_dna_and_thinned(name, dnahist: np.ndarray, ThinnedDict):
    """
    Takes in a DNA dataset, 
    works out the best match thinned plot and then plots the DNA dataset with the best match thinned plot
    :param dnahist: A numpy array containing the histogram of the normalised Interorigin spacing.
    :param EigenDis: Directory containing the eigenvalues for different p values.
    """
    min_p, min_error = min_p_searcher(dnahist, ThinnedDict, name)
    # Lower = 0           # Histogram lower bound
    # Upper = 3           # Histogram upper bound
    # binno = 50          # Number of histogram bins

    min_p = round(min_p, 2)  # Round p to 2 decimal places
    # Area under dnahist should be 1
    # bin_width = np.diff(bins)[0]
    # area = np.sum(dnahist * bin_width)
    # dnahist = dnahist / area
    # newarea = np.sum(dnahist * bin_width)
    # assert np.isclose(newarea, 1), "Area under the dnahist is not 1"
    #print(f"Area under dnahist: {newarea}")
    bins = ThinnedDict["Bins"]
    bincentres = 0.5 * (bins[:-1] + bins[1:])
    thinnedhist = ThinnedDict[min_p]
    bin_width = np.diff(bins)[0]

    exp = ThinnedDict[0.00]
    #print(ThinnedDict.keys())
    wig = ThinnedDict[1.00]

    dnahist = dnahist / np.sum(dnahist * bin_width)
    thinnedhist = thinnedhist / np.sum(thinnedhist * bin_width)

    # dna pdf only
    plt.figure(figsize=(8, 6))
    plt.plot(bincentres, dnahist, color='blue', lw=2, label=f"Normalised Interorigin Spacing for {name}")
    plt.xlabel(f"Normalised Spacing", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    #plt.title(f"Normalised DNA Spacing for {name}", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DNAPlots/DNA_{name}.png', dpi=300)
    plt.show()

    # thin vs dna pdf
    plt.figure(figsize=(8, 6))
    plt.plot(bincentres, dnahist, color='blue', lw=2, label=f"Normalised Interorigin Spacing for {name}")
    plt.plot(bincentres, thinnedhist, color='red', lw=2, label=f"Thinned Eigenvalues (p={min_p:.2f})")
    plt.xlabel(f"Normalised Spacing", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    #plt.title(f"Normalised DNA Spacing for {name} and Thinned Wigner's Surmise", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DNAPlots/DNA_vs_Thinned_{name}.png', dpi=300)
    plt.show()

    # Exp, Wigner, DNA
    plt.figure(figsize=(8, 6))
    plt.plot(bincentres, exp, color='green', lw=2, label=f"Exponential Distribution")
    plt.plot(bincentres, dnahist, color='blue', lw=2, label=f"Normalised Interorigin Spacing for {name}")
    plt.plot(bincentres, wig, color='red', lw=2, label=f"Wigner's Surmise")
    plt.xlabel(f"Normalised Spacing", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    #plt.title(f"Normalised DNA Spacing for {name} and Exponential Distribution", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DNAPlots/DNA_vs_Exponential_{name}.png', dpi=300)
    plt.show()

    # CDFs
    dna_cdf = np.cumsum(dnahist * bin_width)
    thinned_cdf = np.cumsum(thinnedhist * bin_width)
    cdf_RMSE = np.sqrt(np.sum((dna_cdf - thinned_cdf)**2)/len(dna_cdf))

    plt.figure(figsize=(8, 6))
    plt.plot(bincentres, dna_cdf, color='blue', lw=2, label=f"Normalised Interorigin Spacing for {name}")
    plt.plot(bincentres, thinned_cdf, color='red', lw=2, label=f"Thinned Eigenvalues (p={min_p:.2f}, RMSE={cdf_RMSE:.3f})")
    plt.xlabel(f"Normalised Spacing", fontsize=12)
    plt.ylabel("Cumulative Density", fontsize=12)
    #plt.title(f"Normalised DNA Spacing for {name} and Thinned Wigner's Surmise", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DNAPlots/DNA_vs_Thinned_{name}Cumulative.png', dpi=300)
    plt.show()

    exp_cdf = np.cumsum(exp * bin_width)
    wig_cdf = np.cumsum(wig * bin_width)
    plt.figure(figsize=(8, 6))
    plt.plot(bincentres, exp_cdf, color='green', lw=2, label=f"Exponential Distribution")
    plt.plot(bincentres, dna_cdf, color='blue', lw=2, label=f"Normalised Interorigin Spacing for {name}")
    plt.plot(bincentres, wig_cdf, color='red', lw=2, label=f"Wigner's Surmise")
    plt.xlabel(f"Normalised Spacing", fontsize=12)
    plt.ylabel("Cumulative Density", fontsize=12)
    #plt.title(f"Normalised DNA Spacing for {name} and Exponential Distribution", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DNAPlots/DNA_vs_Exponential_{name}Cumulative.png', dpi=300)
    plt.show()


    
