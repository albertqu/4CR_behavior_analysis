# system
import os, re
from numbers import Number
# data
import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# PRINCIPLE: CODE: 0 INDEX, PLOTTING: 1 INDEX
""" ##########################################
################ Preprocessing ###############
########################################## """


def load_session(root, opt='all'):
    file_map = {'MAS': "WilbrechtLab_4ChoiceRevDataALL_MASTERFILE_081513.csv",
                'FIP': "FIP Digging Summary Master File.csv"}
    if isinstance(opt, str):
        if opt == 'all':
            opts = file_map.keys()
        else:
            opts = []
            assert opt in file_map, 'dataset not found'
            return pd.read_csv(os.path.join(root, file_map[opt]))
    else:
        opts = opt
    pdfs = [pd.read_csv(os.path.join(root, file_map[o])) for o in opts]
    return pd.concat(pdfs, axis=0, join='outer', ignore_index=True)


def label_feature(pdf, criteria='num', verbose=False):
    # OBSOLETE
    results = []
    for c in pdf.columns:
        try:
            cdata = pdf[c]
            if criteria == 'num':
                l = int(np.issubdtype(cdata.dtype, np.number) or np.all(cdata.dropna().str.isnumeric()))
            elif criteria == 'full':
                l = int(not np.any(cdata.isnull()))
            else:
                raise NotImplementedError('Unknown criteria: ', criteria)
        except:
            l = -1
        results.append(l)
        if verbose:
            print(c, l)
    return np.array(results)


def dataset_feature_cleaning(pdf, spec_map):
    # TODO: implement merger
    # TODO: maybe at some point drop rows instead of slicing them to rid the warnings?
    # TODO: move function outside to form a new function with a new table
    # Search for specific selectors
    for s in spec_map:
        # TODO: add function to allow multiplexing handling
        if spec_map[s] == 'selD':
            # Select rows with nonnull values
            pdf = pdf.loc[~pdf[s].isnull()]
            del spec_map[s]
        elif spec_map[s] == 'selF':
            pdf.loc[pdf[s].isnull(), s] = -1
            pdf[s] = pdf[s].astype(np.int)
            del spec_map[s]
        elif isinstance(spec_map[s], dict):
            # Select
            assert 'sel' in spec_map[s], print('Bad Usage of Dictionary Coding, no drop nor sel')
            sel_list = spec_map[s]['sel']
            pdf = pdf.loc[pdf[s].isin(sel_list)]
            del spec_map[s]
    return pdf


def vanilla_vectorizer(pdf):
    keywords = pdf.columns[label_feature(pdf) == 1][:-2]
    X0 = pdf[keywords].astype(np.float64)
    X_nonan = X0.dropna()
    return X_nonan


def MAS_data_vectorizer(pdf, inputRange, NAN_policy='drop'):
    """

    :param pdf:
    :param inputRange: list (denoting the column indices) or int (first inputRange columns are inputs)
    :param NAN_policy:
    :return:
    """
    spec_map = {
        'exp1_label_FI_AL_M': 'selF',
        'exp2_Angel': 'selF',
        'Treat': 'cat',
        'Age At Treat': 'cat',
        'Genotype': 'cat',
        'Experimenter': 'cat'
    }
    # test.weight
    testweight = pdf['test.weight']
    for i in range(testweight.shape[0]):
        target = testweight.iloc[i]
        if ' at ' in str(target):
            dp = float(target.split(" ")[0])
            pdf.loc[i, 'test.weight'] = dp

    if not isinstance(inputRange, list):
        inputRange = np.arange(inputRange)
    outputRange = np.setdiff1d(np.arange(pdf.shape[-1]), inputRange)
    pdf = dataset_feature_cleaning(pdf, spec_map)
    inPDF, outPDF = pdf.iloc[:, inputRange], pdf.iloc[:, outputRange]
    tPDF, tLabels, tFRows = default_vectorizer(inPDF, NAN_policy, spec_map, True)
    xPDF, xLabels, xFRows = default_vectorizer(outPDF, NAN_policy, spec_map, True)
    mRows = tFRows & xFRows
    return (tPDF.loc[mRows], tLabels.loc[mRows]), (xPDF.loc[mRows], xLabels.loc[mRows])


def FIP_data_vectorizer(pdf, inputRange, NAN_policy='drop'):
    """
    :param pdf:
    :param inputRange: list (denoting the column indices) or int (first inputRange columns are inputs)
    :param NAN_policy:
    :return:
    """
    spec_map = {
        'Experimental Cohort': 'cat',
        'FIP Duration': 'cat',
        'Behavior Duration': 'cat',
    }
    if not isinstance(inputRange, list):
        inputRange = np.arange(inputRange)
    outputRange = np.setdiff1d(np.arange(pdf.shape[-1]), inputRange)
    inPDF, outPDF = pdf.iloc[:, inputRange], pdf.iloc[:, outputRange]
    tPDF, tLabels, tFRows = default_vectorizer(inPDF, NAN_policy, spec_map, True)
    xPDF, xLabels, xFRows = default_vectorizer(outPDF, NAN_policy, spec_map, True)
    mRows = tFRows & xFRows
    return (tPDF.loc[mRows], tLabels.loc[mRows]), (xPDF.loc[mRows], xLabels.loc[mRows])


def default_vectorizer(pdf, NAN_policy='drop', spec_map=None, finalROW=False):
    """ Taking in pdf with raw data, returns tuple(PDF_feature, encoding PDF)
    Every modification to rows must be recorded if the features are to be used in a (Input, Output) model
    and finalRow must be `True`.
    # TODO: when setting data values, use pdf.values[slice, slice] = value
    :param pdf:
    :return:
    """
    # NAN_policy: fill additional value (minimum for instance)
    if spec_map is None:
        spec_map = {}

    # Check drop
    droplist = [c for c in pdf.columns if c in spec_map and spec_map[c] == 'drop']
    pdf.drop(columns=droplist, inplace=True)

    class_table = classify_features(pdf)
    # class 2, special treatment out front
    SPECIAL_MAP = class_table['class'] == 2
    special_features = class_table['feature'][SPECIAL_MAP]
    # ONLY special features that got converted to other types will be kept in the final data frame
    for spe in special_features:
        sdata = pdf[spe]
        null_flags = sdata.isnull()
        nonnull = sdata[~null_flags]
        m = re.search("(\d+)/(\d+)/(\d+)", nonnull.iloc[0])
        if m:
            # TODO: if some relationship were to be found, we could split date code to three column year,
            #  month, date to probe seasonality
            class_table['class'][class_table['feature'] == spe] = 1
            for i in range(sdata.shape[0]):
                if not null_flags[i]:
                    m = re.search("(\d+)/(\d+)/(\d+)", sdata.iloc[i])
                    g3 = m.group(3)
                    if len(g3) == 4:
                        g3 = g3[-2:]
                    datecode = int(f"{int(g3):02d}{int(m.group(1)):02d}{int(m.group(2)):02d}")
                    pdf.loc[i, spe] = datecode
                else:
                    pdf.loc[i, spe] = np.nan

        elif spe in spec_map:
            class_table['class'][class_table['feature'] == spe] = 0
            if spec_map[spe] == 'cat':
                del spec_map[spe]
        else:
            for i, s in enumerate(nonnull):
                try:
                    if str(s).startswith("#"):
                        try:
                            pdf[spe].replace({s: np.nan}, inplace=True)
                            pdf[spe] = pdf[spe].astype(np.float)
                            class_table['class'][class_table['feature'] == spe] = 1
                        except:
                            print(f'feature {spe} needs special handling')
                        break
                except:
                    print(spe, i, 'nonnull')
                    raise RuntimeError()

    # class 0, encode, use hot map
    ALPHAS_MAP = class_table['class'] == 0
    alpha_features = class_table['feature'][ALPHAS_MAP]
    pdf_ENC = pdf[alpha_features].copy()
    for alp in alpha_features:
        if alp in spec_map:
            # spec_map features demand special encoding
            assert isinstance(spec_map[alp], dict), 'special map must be a dictionary'
            pdf[alp].replace(spec_map[alp], inplace=True)
        else:
            null_alp = pdf[alp].isnull()
            unique_vals = pdf[alp].dropna().unique()
            if len(unique_vals) < 3:
                # Default binary coding
                pdf[alp]=pdf[alp].astype('category')
                pdf_ENC[alp] = pdf[alp]
                pdf[alp] = pdf[alp].cat.codes.astype(np.float)
                # TODO: clean the setter algorithms by converting everything to normal
                pdf.loc[null_alp, alp] = np.nan
            else:
                # ONE HOT ENCODING if more than 2 possible values
                # NULL value be sure to mark back to null
                pdf = pd.get_dummies(pdf, prefix_sep='__', columns=[alp], dtype=np.float)
                pdf.loc[null_alp, [c for c in pdf.columns if c.startswith(alp+'__')]] = np.nan
            # if NAN greater than 50%, apply NAN_policy first

    NUM_MAP = class_table['class'] == 1
    num_features = class_table['feature'][NUM_MAP]
    pdf.drop(columns=np.setdiff1d(class_table['feature'],
                                        np.concatenate((alpha_features, num_features))), inplace=True)

    pdf = pdf.astype(np.float)
    # DISPOSE OF columns that have too many nans
    if NAN_policy == 'drop':
        THRES = 0.3
        dropped_cols = class_table['null_ratio'] >= THRES
        dropped_features = class_table['feature'][dropped_cols]
        dropped_alpha_features = class_table['feature'][ALPHAS_MAP & dropped_cols]
        pdf_ENC.drop(columns=dropped_alpha_features, inplace=True)
        droppingFs = []
        for feat in dropped_features:
            if feat in pdf.columns:
                droppingFs.append(feat)
            else:
                droppingFs = droppingFs + [c for c in pdf.columns if c.startswith(feat+'__')]
        pdf.drop(columns=droppingFs, inplace=True)
        nonullrows = ~pdf.isnull().any(axis=1)
    if finalROW:
        return pdf, pdf_ENC, nonullrows
    else:
        pdf = pdf.loc[nonullrows]
        pdf_ENC = pdf_ENC.loc[nonullrows]
        return pdf, pdf_ENC


def categorical_feature_num_to_label(pdf, pdfENC, column):
    orig, val = column.split('__')
    return pdfENC[orig]


def classify_features(pdf, out=None):
    # cat: 0, num: 1, special: 2
    # out: tuple(folder, saveopt)
    class_table = {}
    class_table['feature'] = []
    class_table['class'] = []
    class_table['null_ratio'] = []
    for c in pdf.columns:
        class_table['feature'].append(c)
        cdata = pdf[c]
        null_sels = cdata.isnull()
        null_ratio = null_sels.sum() / len(cdata)
        nonnulls = cdata[~null_sels]
        if np.issubdtype(cdata.dtype, np.number) or np.all(nonnulls.str.isnumeric()):
            class_table['class'].append(1)
        elif np.all(nonnulls.str.isalpha()):
            class_table['class'].append(0)
        else:
            class_table['class'].append(2)
        class_table['null_ratio'].append(null_ratio)
    for k in class_table:
        class_table[k] = np.array(class_table[k])
    if out is not None:
        outfolder, saveopt = out
        pd.DataFrame(class_table).to_csv(os.path.join(outfolder, f'{saveopt}_classTable.csv'), index=False)
    return class_table


""" ##########################################
################ Visualization ###############
########################################## """


def visualize_dimensionality_ratios(pca_full, dataset_name, JNOTEBOOK_MODE=False):
    xs = np.arange(1, len(pca_full.singular_values_)+1)
    ys = np.cumsum(pca_full.explained_variance_ratio_)
    titre = dataset_name+' PCA dimension plot'
    if JNOTEBOOK_MODE:
        px.line(pd.DataFrame({'components': xs, 'variance ratio': ys}),
                x='components', y='variance ratio', title=titre)
    else:
        plt.plot(xs, ys, 'o-')
        plt.xlabel('# Components')
        plt.ylabel("cumulative % of variance")
        plt.title(titre)
        plt.show()


def visualize_loading_weights(pca, keywords, nth_comp, show=True):
    xs = np.arange(keywords.shape[0])
    loadings = pca.components_[nth_comp]
    pos = loadings >= 0
    plt.bar(xs[pos], loadings[pos], color='b', label='+weights')
    plt.bar(xs[~pos], -loadings[~pos], color='r', label='-weights')
    plt.xticks(xs, keywords, rotation=90)
    plt.legend()
    plt.subplots_adjust(bottom=0.3)
    if show:
        plt.show()


def save_loading_plots(decomp, keywords, model_name, plots):
    outpath = os.path.join(plots, f'{model_name}_numerical_features')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for i in range(decomp.components_.shape[0]):
        plt.figure(figsize=(15, 8))
        visualize_loading_weights(decomp, keywords, i, show=False)
        fname = os.path.join(outpath, f'{model_name}_numF_D{i+1}_loading')
        plt.savefig(fname+'.png')
        plt.savefig(fname+'.eps')
        plt.close()


def visualize_2D(X_HD, labels, dims, tag, show=True, out=None, JNOTEBOOK_MODE=False):
    # TODO: implement saving for JNOTEBOOK_MODE
    """dims: 0 indexed"""
    # change clustering
    nlabels = np.unique(labels)
    if not hasattr(dims, '__iter__'):
        dims = [dims, dims + 1]
    X_2D = X_HD[:, dims]
    c0n = f'Comp {dims[0]+1}'
    c1n = f'Comp {dims[1]+1}'
    titre = f'{tag}_2D_{dims[0]}-{dims[1]}_vis'
    if JNOTEBOOK_MODE:
        fig = px.scatter(pd.DataFrame({c0n: X_2D[:, 0], c1n: X_2D[:, 1],
                                 labels.name: labels.values}), x=c0n, y=c1n, color=labels.name, title=titre)
        fig.show()
    else:
        PALETTE = sns.color_palette("hls", len(nlabels))
        fig = plt.figure(figsize=(15, 15))
        for i, l in enumerate(nlabels):
            print('label', l)
            plt.scatter(X_2D[labels == l, 0], X_2D[labels == l, 1], color=PALETTE[i], label=l)
        plt.legend()
        plt.title(f'{tag} 2D {dims} visualization')
        plt.xlabel(c0n)
        plt.ylabel(c1n)
        if show:
            plt.show()
        if out is not None:
            fname = os.path.join(out, titre)
            plt.savefig(fname+'.png')
            plt.savefig(fname+'.eps')
            plt.close()


def save_LD_plots(X_LD, labels, tag, out, show=True):
    # labels: pd.Series containing labels for different sample points
    outpath = os.path.join(out, tag+"_2D", 'labelGroup_'+labels.name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for i in range(X_LD.shape[1] - 1):
        visualize_2D(X_LD, labels, i, tag, show=show, out=outpath)


def visualize_3D(X_HD, labels, dims, tag, show=True, out=None, JNOTEBOOK_MODE=False):
    # TODO: implement saving for JNOTEBOOK_MODE
    """dims: 0 indexed"""
    # change clustering
    nlabels = np.unique(labels)
    if not hasattr(dims, '__iter__'):
        dims = np.arange(dims, dims+3)
    assert X_HD.shape[1] >= 3, 'not enough dimensions try 2d instead'
    X_2D = X_HD[:, dims]
    c0n = f'Comp {dims[0]+1}'
    c1n = f'Comp {dims[1]+1}'
    c2n = f'Comp {dims[2]+1}'
    titre = f'{tag}_3D_{dims[0]}-{dims[1]}-{dims[2]}_vis'
    if JNOTEBOOK_MODE:
        fig = px.scatter_3d(pd.DataFrame({c0n: X_2D[:, 0], c1n: X_2D[:, 1], c2n: X_2D[:, 2],
                                 labels.name: labels.values}), x=c0n, y=c1n, color=labels.name, title=titre)
        fig.show()
    else:
        raise NotImplementedError('3D is currently only supported in Jupiter Notebooks')
        # PALETTE = sns.color_palette("hls", len(nlabels))
        # fig = plt.figure(figsize=(15, 15))
        # for i, l in enumerate(nlabels):
        #     print('label', l)
        #     plt.scatter(X_2D[labels == l, 0], X_2D[labels == l, 1], color=PALETTE[i], label=l)
        # plt.legend()
        # plt.title(f'{tag} 3D {dims} visualization')
        # plt.xlabel(c0n)
        # plt.ylabel(c1n)
        # if show:
        #     plt.show()
        # if out is not None:
        #     fname = os.path.join(out, titre)
        #     plt.savefig(fname+'.png')
        #     plt.savefig(fname+'.eps')
        #     plt.close()


def feature_clustering_visualize(X, keywords, nclusters, show=True):
    agglo = cluster.FeatureAgglomeration(n_clusters=nclusters)
    agglo.fit(X)
    ys = np.arange(keywords.shape[0])
    labels = agglo.labels_
    sorts = np.argsort(labels)
    plt.barh(ys, labels[sorts])
    plt.yticks(ys, keywords[sorts])
    plt.subplots_adjust(left=0.3)
    if show:
        plt.show()
    return agglo


#deprecated
def save_isomap_2D(iso, X_iso, plots, clustering=None):
    if not os.path.exists(plots):
        os.makedirs(plots)
    for i in range(X_iso.shape[1]-1):
        plt.figure(figsize=(15, 15))
        if clustering is not None:
            for l in np.unique(clustering.labels_):
                plt.scatter(X_iso[clustering.labels_ == l, i], X_iso[clustering.labels_ == l, i+1], label=l)
            plt.legend()
        else:
            plt.scatter(X_iso[:, i], X_iso[:, i+1])
        plt.xlabel(f'Comp {i+1}')
        plt.ylabel(f'Comp {i+2}')
        fname = os.path.join(plots, f'iso_2D{i+1}-{i+2}_loading' + '_cluster' if clustering else '')
        plt.savefig(fname+'.png')
        plt.savefig(fname+'.eps')
        plt.close()


""" ##########################################
################# Analysis ###################
########################################## """
# Dim Reduction


# Regression


# Model Testing
def test_model_efficacy(model, X_train, y_train, X_test, y_test):
    train_accus, test_accus = np.empty(y_train.shape[1]), np.empty(y_train.shape[1])
    for i in range(y_train.shape[1]):
        model.fit(X_train, y_train[:, i])
        train_accu = model.score(X_train, y_train[:, i])
        test_accu = model.score(X_test, y_test[:,i])
        train_accus[i] = train_accu
        test_accus[i] = test_accu
    return train_accus, test_accus


""" ##########################################
#################### MAIN ####################
########################################## """


def main():
    # TODO: use grid search to build a great regression model
    # TODO: 3D visualization
    # TODO: add a drop category for vectorizer such that certain features can be omitted
    # TODO: add 95.5 PC variance threshold
    # TODO: add function to handle merged datasets
    # paths
    dataRoot = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/PCA_adversity/data"
    dataset_name = 'FIP' # 'MAS'
    JNOTEBOOK_MODE = False
    CAT_ENCODE = None
    NAN_POLICY = 'drop'

    dataOut = os.path.join(dataRoot, dataset_name)
    plots = os.path.join("/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/PCA_adversity/plots/",
                         dataset_name)
    if not os.path.exists(plots):
        os.makedirs(plots)

    if not os.path.exists(dataOut):
        os.makedirs(dataOut)
    pdf = load_session(dataRoot, dataset_name)

    # preprocessing
    RANGES = {'FIP': 14,  # Behavior only
              'MAS': list(range(6))+list(range(pdf.shape[1]-8, pdf.shape[1])),
              'all': 14} # fix for all
    vectorize_func = {
        'FIP': FIP_data_vectorizer,
        'MAS': MAS_data_vectorizer
    }

    (tPDF, tLabels), (xPDF, xLabels) = vectorize_func[dataset_name](pdf, RANGES[dataset_name], NAN_POLICY)
    tLEN, xLEN = tPDF.shape[1], xPDF.shape[1]
    keywordsT, keywordsX = tPDF.columns, xPDF.columns
    # Takes in demeaned data matrix (np.ndarray), and starts analysis, TODO: obtain X_cat_map, and X_cat labels
    X_nonan_dm = xPDF.sub(xPDF.mean()).values
    T_nonan_dm = tPDF.sub(tPDF.mean()).values

    # DIM Probing
    pca_full = PCA().fit(X_nonan_dm) # N x D -> N x K * K x D
    X_HD = X_nonan_dm @ (pca_full.components_.T)
    visualize_dimensionality_ratios(pca_full, dataset_name, JNOTEBOOK_MODE=JNOTEBOOK_MODE)
    ClusteringDim = 3 # Determine with the ratio plots MAS: 9
    print(f"Capturing {100*np.sum(pca_full.explained_variance_ratio_[:ClusteringDim]):.4f}% variance")

    # Correlation Matrix
    if T_nonan_dm.shape[1] < X_nonan_dm.shape[1]:
        fLen = X_nonan_dm.shape[1]
        sLen = T_nonan_dm.shape[1]
        fDM, sDM = X_nonan_dm, T_nonan_dm
        fKWs, sKWs = keywordsX, keywordsT
    else:
        fLen = T_nonan_dm.shape[1]
        sLen = X_nonan_dm.shape[1]
        fDM, sDM = T_nonan_dm, X_nonan_dm
        fKWs, sKWs = keywordsT, keywordsX
    corrMat = np.zeros((fLen, sLen))
    for i in range(fLen):
        for j in range(sLen):
            corrMat[i, j] = np.corrcoef(fDM[:, i], sDM[:, j])[0, 1]
    plt.figure(figsize=(15, 15))
    plt.imshow(corrMat ** 2)
    plt.colorbar()
    plt.yticks(np.arange(len(fKWs)), fKWs)
    plt.xticks(np.arange(len(sKWs)), sKWs, rotation=90)
    plt.subplots_adjust(left=0.3, right=0.7, top=0.99, bottom=0.35)
    plt.show()
    corrPDF = pd.DataFrame(data=corrMat, index=fKWs, columns=sKWs)
    if JNOTEBOOK_MODE:
        corrPDF.head()
    else:
        corrPDF.to_csv(os.path.join(dataOut, dataset_name+'corrMatrix.csv'), index=True)

    # DIM Reduction
    # MAS: ['Genotype', 'Treat']
    LDLabels = tLabels['Treat']
    #LDLabels = tLabels['Group'] # Experimental Cohort, FIP Duration, Behavior Duration, Group, Sex
    #LDLabels = tLabels['Sex']
    # PCA
    # save_LD_plots(X_HD, LDLabels, 'PCA', plots, show=False)
    visualize_2D(X_HD, LDLabels, 0, 'PCA', show=True, JNOTEBOOK_MODE=JNOTEBOOK_MODE)
    # ICA
    fastICA = FastICA(n_components=ClusteringDim, random_state=0)
    X_ICAHD = fastICA.fit_transform(X_nonan_dm)
    # save_LD_plots(X_ICAHD, LDLabels, 'ICA', plots, show=False)
    visualize_2D(X_ICAHD, LDLabels, 0, 'ICA', show=True, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # ISOMAP
    embedding = Isomap(n_components=ClusteringDim)
    X_isomap = embedding.fit_transform(X_nonan_dm)
    # save_LD_plots(X_isomap, LDLabels, 'ISOMAP', plots, show=False)
    visualize_2D(X_isomap, LDLabels, 0, 'ISOMAP', show=True, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # Kernel PCA
    kernelPCA = KernelPCA(n_components=ClusteringDim*2, kernel='poly')
    X_kernel = kernelPCA.fit_transform(X_nonan_dm)
    #save_LD_plots(X_kernel, LDLabels, 'Kernel_PCA', plots, show=False)
    visualize_2D(X_kernel, LDLabels, 0, 'Kernel_PCA', show=True, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # ------------------------------- LOOP -------------------------------
    for l in tLabels.columns:
        LDLabels = tLabels[l]
        # PCA
        save_loading_plots(pca_full, keywordsX, 'PCA', plots)
        save_LD_plots(X_HD, LDLabels, 'PCA', plots, show=False)
        # ICA
        fastICA = FastICA(n_components=ClusteringDim, random_state=0)
        X_ICAHD = fastICA.fit_transform(X_nonan_dm)
        save_loading_plots(fastICA, keywordsX, 'ICA', plots)
        save_LD_plots(X_ICAHD, LDLabels, 'ICA', plots, show=False)

        # ISOMAP
        embedding = Isomap(n_components=ClusteringDim)
        X_isomap = embedding.fit_transform(X_nonan_dm)
        save_LD_plots(X_isomap, LDLabels, 'ISOMAP', plots, show=False)

        # Kernel PCA
        kernelPCA = KernelPCA(n_components=ClusteringDim * 2, kernel='poly')
        X_kernel = kernelPCA.fit_transform(X_nonan_dm)
        save_LD_plots(X_kernel, LDLabels, 'Kernel_PCA', plots, show=False)

    # ------------------------------------------------------------------------

    # Clustering
    X_LD = X_HD[:, :ClusteringDim]
    clustering = SpectralClustering(n_clusters=2,
                                    assign_labels="discretize",
                                    random_state=0).fit(X_LD)

    # Regression
    X_train, X_test, y_train, y_test = train_test_split(
        T_nonan_dm, X_nonan_dm, test_size=0.3, random_state=0)
    # Linear Regression
    reg = LinearRegression().fit(X_train, y_train)
    composite_linreg = reg.score(X_test, y_test)
    train_accus, test_accus = test_model_efficacy(reg, X_train, y_train, X_test, y_test)
    errPDF = pd.DataFrame({'train_accuracy': train_accus, 'test_accuracy': test_accus}, index=keywordsX)
    errPDF.to_csv(os.path.join(dataOut, dataset_name + '_linreg_err.csv'), index=True)

    # Decision Tree Regression
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    composite_treereg = regressor.score(X_test, y_test)
    plt.bar(np.arange(len(keywordsT)), regressor.feature_importances_)
    plt.xticks(np.arange(len(keywordsT)), keywordsT, rotation=90)
    plt.subplots_adjust(left=0.3, right=0.7, top=0.99, bottom=0.35)
    plt.show()
    train_accu, test_accus = test_model_efficacy(regressor, X_train, y_train, X_test, y_test)
    errPDF = pd.DataFrame({'train_accuracy': train_accus, 'test_accuracy': test_accus}, index=keywordsX)
    errPDF.to_csv(os.path.join(dataOut, dataset_name + '_DT_regressor_err.csv'), index=True)

    kernelType = 'poly' #['linear', 'poly', 'rbf', 'sigmoid']
    clf = SVR(C=1, epsilon=0.5, kernel=kernelType)
    train_errors, test_errors = test_model_efficacy(clf, X_train, y_train, X_test, y_test)
    errPDF = pd.DataFrame({'train_err': train_errors, 'test_err': test_errors}, index=keywordsX)







