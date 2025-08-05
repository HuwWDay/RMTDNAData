import cmath
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
from dppy.beta_ensembles import CircularEnsemble
import matplotlib.pyplot as plt

# Code for generating thinned Wigner's 

circular = CircularEnsemble(beta=1)

def EigenGen(Size, Samples):
    """Generate eigenvalues from the Circular Ensemble."""
    Eigen = []
    for j in range(Samples):
        if j % (Samples // 20) == 0:  # Update every 5% progress
            print(f"Progress: {100 * j / Samples:.1f}%")
        X = circular.sample_full_model(size_N=Size, haar_mode='QR')
        Angles = np.sort([(cmath.phase(num) + 2 * cmath.pi) * Size / (2 * cmath.pi) for num in X])
        Eigen.extend(Angles)
    return Eigen

def EigenThin(Eigen, p):
    """Randomly thin the eigenvalues and compute normalized gaps."""
    if len(Eigen) == 0:
        return []
    
    y = np.random.binomial(1, p, len(Eigen))
    NewEigen = [x for idx, x in enumerate(Eigen) if y[idx] == 1]
    NewEigen = np.array(NewEigen)
    
    if len(NewEigen) < 2:
        return []  # Not enough points to form gaps
    
    Gaps = np.diff(NewEigen)
    Gaps = Gaps[Gaps >= 0]
    
    if len(Gaps) == 0:
        return []
    else: 
        Avg = np.mean(Gaps)
        if Avg != 0:
            Gaps = [g / Avg for g in Gaps]
    return Gaps

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
    Thinned = ThinnedDict[p]
    # Make Thinned a numpy array
    #Thinned = Thinned.to_numpy()
    #dnahist = dnahist[:-1]  # Remove the last bin (the right edge of the last bin)
    #print(p)
    # print(len(Thinned), len(dnahist))
    assert len(Thinned) == len(dnahist), f"Length of Thinned and dnahist do not match. Length of Thinned: {len(Thinned)}, Length of dnahist: {len(dnahist)}"
    RMSE = np.sqrt(np.sum((dnahist - Thinned)**2)/len(dnahist))
    #print(f"RMSE for p = {p}: ", RMSE)
    return RMSE

def min_p_searcher(dnahist: np.ndarray, ThinnedDict):
    """
    Load in dna data and search for the minimum p value that fits the data.
    :param dnahist: A numpy array containing the histogram of the normalised Interorigin spacing.
    :param EigenDis: Directory containing the eigenvalues for different p values.
    :return: The minimum p value that fits the data.
    """

    # Coarse search
    p_list = np.arange(0, 1.0, 0.01)
    error_list = []
    for p in p_list:

        rounded_p = round(float(p), 2)  # Ensure p matches dictionary key
        RMSE = dna_fit_p(dnahist, ThinnedDict, rounded_p)
        error_list.append(RMSE)
    
    # Plot the errors for search
    plt.figure(figsize=(10, 6))
    plt.plot(p_list, error_list, marker='o', linestyle='-', color='b')
    plt.show()

    # Find the minimum error and corresponding p value
    min_error = min(error_list)
    min_p = p_list[np.argmin(error_list)]
    print(f"Minimum p value in coarse search: {min_p} with RMSE: {min_error}")

    return min_p, min_error

def plot_dna_and_thinned(name, dnahist: np.ndarray, ThinnedDict):
    """
    Takes in a DNA dataset, 
    works out the best match thinned plot and then plots the DNA dataset with the best match thinned plot
    :param dnahist: A numpy array containing the histogram of the normalised Interorigin spacing.
    :param EigenDis: Directory containing the eigenvalues for different p values.
    """
    min_p, min_error = min_p_searcher(dnahist, ThinnedDict)
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
    thinnedhist = ThinnedDict[min_p]
    bin_width = np.diff(bins)[0]

    dnahist = dnahist / np.sum(dnahist * bin_width)
    thinnedhist = thinnedhist / np.sum(thinnedhist * bin_width)


    plt.figure(figsize=(8, 6))
    plt.plot(bins, dnahist, color='blue', lw=2, label=f"Normalised Interorigin Spacing for {name}")
    plt.plot(bins, thinnedhist, color='red', lw=2, label=f"Thinned Eigenvalues (p={min_p:.2f}, RMSE={min_error:.3f})")
    plt.xlabel(f"Normalised Spacing", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Normalised DNA Spacing for {name} and Thinned Wigner's Surmise", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DNAPlots/DNA_vs_Thinned_{name}.png', dpi=300)
    plt.show()


    # CDFs
    dna_cdf = np.cumsum(dnahist * bin_width)
    thinned_cdf = np.cumsum(thinnedhist * bin_width)
    plt.figure(figsize=(8, 6))
    plt.plot(bins, dna_cdf, color='blue', lw=2, label=f"Normalised Interorigin Spacing for {name}")
    plt.plot(bins, thinned_cdf, color='red', lw=2, label=f"Thinned Eigenvalues (p={min_p:.2f}, RMSE={min_error:.3f})")
    plt.xlabel(f"Normalised Spacing", fontsize=12)
    plt.ylabel("Cumulative Density", fontsize=12)
    plt.title(f"Normalised DNA Spacing for {name} and Thinned Wigner's Surmise, Cumulative Plot", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DNAPlots/DNA_vs_Thinned_{name}Cumulative.png', dpi=300)
    plt.show()


    
