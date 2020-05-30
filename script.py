# system
import os, re
from numbers import Number
# data
import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster, preprocessing
from sklearn.manifold import Isomap
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

RAND_STATE = 2020


# PRINCIPLE: CODE: 0 INDEX, PLOTTING: 1 INDEX
""" ##########################################
################ Preprocessing ###############
########################################## """


def load_session(root, opt='all'):
    file_map = {'MAS': "WilbrechtLab_4ChoiceRevDataALL_MASTERFILE_081513.csv",
                'MAS2': 'WilbrechtLab_4ChoiceRevDataALL_MASTERFILE_051820.csv',
                '4CR': "4CR_Merge_052920.csv",
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


def dataset_split_feature_classes(pdf):
    classes = {'other': []}
    for c in pdf.columns:
        if '.' in c:
            cnames = c.split('.')
            classe = ".".join(cnames[:-1])
            if classe in classes:
                classes[classe].append(c)
            else:
                classes[classe] = [c]
        else:
            classes['other'].append(c)
    return {cl: pdf[classes[cl]] for cl in classes}


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
        elif spec_map[s] == 'selF':
            pdf[s].values[pdf[s].isnull()]= -1
            # pdf[s] = pdf[s].astype(np.int)
        elif isinstance(spec_map[s], dict):
            # Select
            assert 'sel' in spec_map[s], print('Bad Usage of Dictionary Coding, no drop nor sel')
            sel_list = spec_map[s]['sel']
            pdf = pdf.loc[pdf[s].isin(sel_list)]
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
            pdf['test.weight'].values[i] = dp

    if not isinstance(inputRange, list):
        inputRange = np.arange(inputRange)
    D = pdf.shape[-1]
    for irg, rg in enumerate(inputRange):
        if rg < 0:
            inputRange[irg] = D+rg

    outputRange = np.setdiff1d(np.arange(D), inputRange)
    pdf = dataset_feature_cleaning(pdf, spec_map)
    inPDF, outPDF = pdf.iloc[:, inputRange], pdf.iloc[:, outputRange]
    tPDF, tLabels, tFRows = default_vectorizer(inPDF, NAN_policy, spec_map, True)
    xPDF, xLabels, xFRows = default_vectorizer(outPDF, NAN_policy, spec_map, True)
    mRows = tFRows & xFRows
    return (tPDF.loc[mRows], tLabels.loc[mRows]), (xPDF.loc[mRows], xLabels.loc[mRows])


def FOURCR_data_vectorizer(pdf, inputRange, NAN_policy='drop'):
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
        'Genotype': 'cat',
        'Experimenter': 'cat'
    }
    # test.weight
    testweight = pdf['test.weight']
    for i in range(testweight.shape[0]):
        target = testweight.iloc[i]
        if ' at ' in str(target):
            dp = float(target.split(" ")[0])
            pdf['test.weight'].values[i] = dp

    if not isinstance(inputRange, list):
        inputRange = np.arange(inputRange)
    D = pdf.shape[-1]
    for irg, rg in enumerate(inputRange):
        if rg < 0:
            inputRange[irg] = D+rg

    outputRange = np.setdiff1d(np.arange(D), inputRange)
    pdf = dataset_feature_cleaning(pdf, spec_map)
    inPDF, outPDF = pdf.iloc[:, inputRange], pdf.iloc[:, outputRange]
    tPDF, tLabels, tFRows = default_vectorizer(inPDF, 'ignore', spec_map, True)
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
                    pdf[spe].values[i] = datecode
                else:
                    pdf[spe].values[i] = np.nan

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
    pdf_ENC = pdf.loc[:, alpha_features].copy(deep=True)
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
                pdf.loc[:, alp] = pdf[alp].cat.codes.astype(np.float)
                # TODO: clean the setter algorithms by converting everything to normal
                pdf[alp].values[null_alp] = np.nan
            else:
                # ONE HOT ENCODING if more than 2 possible values
                # NULL value be sure to mark back to null
                pdf = pd.get_dummies(pdf, prefix_sep='__', columns=[alp], dtype=np.float)
                pdf[[c for c in pdf.columns if c.startswith(alp+'__')]].values[null_alp] = np.nan
            # if NAN greater than 50%, apply NAN_policy first

    NUM_MAP = class_table['class'] == 1
    num_features = class_table['feature'][NUM_MAP]
    pdf.drop(columns=np.setdiff1d(class_table['feature'],
                                        np.concatenate((alpha_features, num_features))), inplace=True)

    pdf = pdf.astype(np.float)
    # DISPOSE OF columns that have too many nans TODO: IMPLEMENT IGNORE POLICY OF NANS
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
    elif NAN_policy == 'ignore':
        pdf_ENC[pdf_ENC.isnull()] = np.nan
        pdf[pdf.isnull()] = np.nan
        nonullrows = np.full(pdf.shape[0], 1, dtype=bool)
    else:
        raise NotImplementedError(f'unknown policy {NAN_policy}')
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


def get_data_label(tPDF, tLabels, label=None, STRIFY=True):
    # TODO: add merge label function if needed
    if label is None:
        allLabels = tLabels.columns
        return set(np.concatenate([allLabels, [c for c in tPDF.columns if '__' not in c]]))
    else:
        targetDF = None
        if label in tLabels.columns:
            targetDF = tLabels
        elif label in tPDF.columns:
            targetDF = tPDF
        else:
            raise RuntimeError(f'Unknown Label: {label}!')
        LDLabels = targetDF[label]
        if STRIFY:
            LDLabels = LDLabels.astype('str')
        return LDLabels


""" ##########################################
################ Visualization ###############
########################################## """


def visualize_dimensionality_ratios(pca_full, dataset_name, JNOTEBOOK_MODE=False):
    xs = np.arange(1, len(pca_full.singular_values_)+1)
    ys = np.cumsum(pca_full.explained_variance_ratio_)
    titre = dataset_name+' PCA dimension plot'
    if JNOTEBOOK_MODE:
        fig = px.line(pd.DataFrame({'components': xs, 'variance ratio': ys}),
                x='components', y='variance ratio', title=titre)
        fig.show()
    else:
        plt.plot(xs, ys, 'o-')
        plt.xlabel('# Components')
        plt.ylabel("cumulative % of variance")
        plt.title(titre)
        plt.show()


def dataset_dimensionality_probing(X_nonan_dm, dataset_name, visualize=True, verbose=True,
                                   JNOTEBOOK_MODE=False):
    # DIM Probing
    pca_full = PCA().fit(X_nonan_dm) # N x D -> N x K * K x D
    if visualize:
        visualize_dimensionality_ratios(pca_full, dataset_name, JNOTEBOOK_MODE=JNOTEBOOK_MODE)
    #automatic_dimension probing
    # TODO: IMPLEMENT CROSS-VALIDATION BASED SELECTION
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ClusteringDim = np.where(cumsum >= 0.9)[0][0]+1 # Determine with the ratio plots MAS: 9
    variance_ratio = np.sum(pca_full.explained_variance_ratio_[:ClusteringDim])
    if verbose:
        print(f"Capturing {100*variance_ratio:.4f}% variance with {ClusteringDim} components")
    return pca_full, ClusteringDim


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


def visualize_2D(X_HD, labels, dims, tag, show=True, out=None, label_alias=None, JNOTEBOOK_MODE=False):
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
        nlabels = len(labels.unique())
        # TODO: blend in real labels of treatments
        if labels.dtype == 'object':
            cseq = sns.color_palette("coolwarm", nlabels).as_hex()
            cmaps = {ic: cseq[i] for i, ic in enumerate(sorted(labels.unique()))}
            if label_alias is not None:
                newLabels = labels.copy(deep=True)
                newCMAPs = {}
                for l in labels.unique():
                    al = label_alias[l]
                    newLabels[labels == l] = al
                    newCMAPs[al] = cmaps[l]
                labels = newLabels
                cmaps = newCMAPs
        else:
            cmaps = None
        fig = px.scatter(pd.DataFrame({c0n: X_2D[:, 0], c1n: X_2D[:, 1],
                                 labels.name: labels.values}), x=c0n, y=c1n, color=labels.name,
                                 color_discrete_map=cmaps, title=titre)
        fig.show()
    else:
        PALETTE = sns.color_palette("hls", len(nlabels))
        fig = plt.figure(figsize=(15, 15))
        for i, l in enumerate(nlabels):
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


def visualize_3D(X_HD, labels, dims, tag, show=True, out=None, label_alias=None, JNOTEBOOK_MODE=False):
    # TODO: implement saving for JNOTEBOOK_MODE
    """dims: 0 indexed"""
    # change clustering
    if not hasattr(dims, '__iter__'):
        dims = np.arange(dims, dims+3)
    assert X_HD.shape[1] >= 3, 'not enough dimensions try 2d instead'
    X_2D = X_HD[:, dims]
    c0n = f'Comp {dims[0]+1}'
    c1n = f'Comp {dims[1]+1}'
    c2n = f'Comp {dims[2]+1}'
    titre = f'{tag}_3D_{dims[0]}-{dims[1]}-{dims[2]}_vis'

    if JNOTEBOOK_MODE:
        nlabels = len(labels.unique())
        # TODO: blend in real labels of treatments
        if labels.dtype == 'object':
            cseq = sns.color_palette("coolwarm", nlabels).as_hex()
            cmaps = {ic: cseq[i] for i, ic in enumerate(sorted(labels.unique()))}
            if label_alias is not None:
                newLabels = labels.copy(deep=True)
                newCMAPs = {}
                for l in labels.unique():
                    al = label_alias[l]
                    newLabels[labels == l] = al
                    newCMAPs[al] = cmaps[l]
                labels = newLabels
                cmaps = newCMAPs
        else:
            cmaps = None
        fig = px.scatter_3d(pd.DataFrame({c0n: X_2D[:, 0], c1n: X_2D[:, 1], c2n: X_2D[:, 2],
                                 labels.name: labels.values}), x=c0n, y=c1n, z=c2n,
                            color=labels.name, color_discrete_map=cmaps, title=titre)
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


def visualize_LD_multimodels(models, labels, dims, ND=3, show=True, out=None, label_alias=None,
                             JNOTEBOOK_MODE=False):
    ND = min(list(models.values())[0][1].shape[1], ND)
    vfunc = visualize_3D if ND == 3 else visualize_2D

    for m in models:
        model, X_LD = models[m]
        vfunc(X_LD, labels, dims, m, show=show, out=out, label_alias=label_alias,
              JNOTEBOOK_MODE=JNOTEBOOK_MODE)


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


# Basic
def feature_preprocess(tPDF, xPDF, method='skscale'):
    # TODO: implement more detailed feature preprocessing based on feature relationships
    # in fact tPDF does not need much unit variance scaling, but is implemented here for consistency
    keywordsT, keywordsX = tPDF.columns, xPDF.columns
    # Takes in demeaned data matrix (np.ndarray), and starts analysis, TODO: obtain X_cat_map, and X_cat labels
    if method == 'skscale':
        X_nonan_dm = preprocessing.scale(xPDF.values, axis=0)
        T_nonan_dm = preprocessing.scale(tPDF.values, axis=0)
    else:
        X_nonan_dm = xPDF.sub(xPDF.mean()).values
        T_nonan_dm = tPDF.sub(tPDF.mean()).values
    return T_nonan_dm, keywordsT, X_nonan_dm, keywordsX


def get_corr_matrix(T_nonan_dm, keywordsT, X_nonan_dm, keywordsX, dataset_name, dataOut,
                    JNOTEBOOK_MODE=False):
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
        corrPDF.to_csv(os.path.join(dataOut, dataset_name + 'corrMatrix.csv'), index=True)


# Dim Reduction
def dim_reduction(X_nonan_dm, ClusteringDim, models='all', kernelPCA_params=None):
    """
    Performs dim reduction on data
    :param models: str or list-like
    kernelPCA_params: tuple (dimMUL, kernel)
        if kernelPCA is one of the models in `models`, kernelPCA_params must not be None
    :return: dict {model: (fit_model, transformed_data)} or just one tuple
    """
    # TODO: MDS, T-SNE
    DRs = {}
    EXTRACT = False
    if models=='all':
        models = ['PCA', 'ICA', 'ISOMAP', 'Kernel_PCA']
    elif isinstance(models, str):
        models = [models]
        EXTRACT = False
    # Maybe Use tSNE
    for m in models:
        DRModel = None
        if m == 'PCA':
            DRModel = PCA(n_components=ClusteringDim)
        elif m == 'ICA':
            DRModel = FastICA(n_components=ClusteringDim, random_state=RAND_STATE)
        elif m == 'ISOMAP':
            DRModel = Isomap(n_components=ClusteringDim)
        elif m == 'Kernel_PCA':
            dimMUL, kernel = kernelPCA_params
            DRModel = KernelPCA(n_components=ClusteringDim*dimMUL, kernel=kernel)
        else:
            raise NotImplementedError(f'model {m} not implemented')
        DRs[m] = (DRModel, DRModel.fit_transform(X_nonan_dm))
    if EXTRACT:
        return list(DRs.values())[0]
    return DRs


def classifier_LD_multimodels(models, labels, LD_dim=None, N_iters=100, mode='pred', ignore_labels=None):
    clfs = {}
    confs = {}
    if ignore_labels is None:
        ignore_labels = []
    uniqLabels = np.setdiff1d(labels.unique(), ignore_labels)
    lxs = np.arange(len(uniqLabels))
    NM = 1
    scores = 0
    # Plot a score find best score ?
    for m in models:
        model, X_LD = models[m]
        if LD_dim is None:
            LD_dim = X_LD.shape[-1]
        X_LD = X_LD[:, :LD_dim]
        # TODO: add stratified k fold later
        confmats = np.zeros((len(uniqLabels), len(uniqLabels)))
        neg_selectors = labels.isin(uniqLabels)
        X_LDF, labelsF = X_LD[neg_selectors], labels[neg_selectors]
        for i in range(N_iters):
            X_train, X_test, y_train, y_test = train_test_split(
                X_LDF, labelsF, test_size=0.3, shuffle=True, stratify=labelsF)
            # QDA
            clf = QuadraticDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            preds = clf.predict(X_test)
            conf_mat = confusion_matrix(y_test, preds, labels=uniqLabels, normalize=mode)
            confmats += conf_mat
            scores += score
        confs[m] = {'QDA': confmats / N_iters}
        clfF = QuadraticDiscriminantAnalysis()
        clfF.fit(X_LDF, labelsF)
        clfs[m] = {'QDA': (clfF, scores / N_iters)} # get clf for all data
    fig, axes = plt.subplots(nrows=NM, ncols=len(models), sharex=True, sharey=True)
    # TODO: Handle multiple models
    for i, m in enumerate(confs):
        if NM * len(models) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(confs[m]['QDA'])
        ax.set_title(m)
        ax.set_xticks(lxs)
        ax.set_xticklabels(uniqLabels)
        ax.set_ylabel('truth')
        ax.set_yticks(lxs)
        ax.set_yticklabels(uniqLabels)
        ax.set_ylabel('prediction')

    return clfs, confs


def clustering_LD_multimodels(models, labels, ClusteringDim=None, show=True, out=None):
    # TODO: finish clutsering method
    clusterings = {}
    NClusters = len(labels.unique())

    for m in models:
        model, X_LD = models[m]
        if ClusteringDim is None:
            ClusteringDim = X_LD.shape[-1]
        clustering[m] = {}
        # Default: Spectral
        X_LD = X_LD[:, :ClusteringDim]
        clustering = SpectralClustering(n_clusters=NClusters,
                                        assign_labels="discretize",
                                        random_state=RAND_STATE).fit(X_LD)
        clustering[m]['spectral'] = clustering


# Regression
def regression_with_norm_input(tPDF, xPDF, dataset_name, dataOut, method='linear', feature_importance=True):
    # TODO: use grid search to build a great regression model
    T_nonan_dm, keywordsT, X_nonan_dm, keywordsX = feature_preprocess(tPDF, xPDF)
    X_train, X_test, y_train, y_test = train_test_split(
        T_nonan_dm, X_nonan_dm, test_size=0.3) # TODO: do multiple iterations
    if method == 'all':
        method = ['linear', 'DT', 'SVR_poly', 'SVR_linear', 'SVR_rbf', 'SVR_sigmoid']
    elif isinstance(method, str):
        method = [method]

    if feature_importance and ('DT' not in method):
        method.append('DT')

    for m in method:
        if m == 'linear':
            # Linear Regression
            reg = LinearRegression().fit(X_train, y_train)
            composite_linreg = reg.score(X_test, y_test)
            train_accus, test_accus = test_model_efficacy(reg, X_train, y_train, X_test, y_test)
            errPDF = pd.DataFrame({'train_accuracy': train_accus, 'test_accuracy': test_accus},
                                  index=keywordsX)
        elif m == 'DT':
            # Decision Tree Regression
            regressor = DecisionTreeRegressor(random_state=RAND_STATE)
            regressor.fit(X_train, y_train)
            composite_treereg = regressor.score(X_test, y_test)
            plt.bar(np.arange(len(keywordsT)), regressor.feature_importances_)
            plt.xticks(np.arange(len(keywordsT)), keywordsT, rotation=90)
            plt.subplots_adjust(left=0.3, right=0.7, top=0.99, bottom=0.35)
            plt.show()
            train_accu, test_accus = test_model_efficacy(regressor, X_train, y_train, X_test, y_test)
            errPDF = pd.DataFrame({'train_accuracy': train_accus, 'test_accuracy': test_accus},
                                  index=keywordsX)
        elif re.search(r'SVR_(\w+)', m):
            kernelType = m.split('_')[1] #['linear', 'poly', 'rbf', 'sigmoid']
            clf = SVR(C=1, epsilon=0.5, kernel=kernelType)
            train_errors, test_errors = test_model_efficacy(clf, X_train, y_train, X_test, y_test)
            errPDF = pd.DataFrame({'train_err': train_errors, 'test_err': test_errors}, index=keywordsX)
        errPDF.to_csv(os.path.join(dataOut, dataset_name + f'_{m}_regressor_err.csv'), index=True)


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
########### DATA-SPECIFIC MODULE #############
########################################## """


def get_test_specific_options(test, BLIND=False):
    # Color key Experiment 1: (Groups 5-8 are unusually flexible; group 1,6-8 are different strains than 2-5)
    exp1 = {"-1.0": r"OTHERS",
            "0.0": r'Controls',
            "1.0": r"WT/SAL male P60-90 Bl/6J/CR",
            "2.0": r"FI male P60 Taconic",
            "3.0": r"FR male P60 Taconic",
            "4.0": r"ALG male P60 Taconic",
            "5.0": r"ALS male P60 Taconic",
            "6.0": r"5d COC test at P90 Bl/6CR",
            "7.0": r"BDNF met/met Ron tested at P60",
            "8.0": r"P26 males WT Bl/6CR"}
    # Color Key Experiment 2 data (focusing on angel's mice and bdnf/trkb manipulations) P40-60 ages
    exp2 = {"-1.0": r'OTHERS',
            "1.0": r"Controls VEH/SAL/WT",
            "2.0": r"acute NMPP1pump",
            "3.0": r"chronic NMPP1pump",
            "4.0": r"BDNF Val/Val Ron",
            "5.0": r"P1-23 NMPP1H20",
            "6.0": r"P1-40 NMPP1H20",
            "7.0": r"BDNF Met/Met Ron"}

    TEST_LABEL_ALIAS = {
        'exp1_label_FI_AL_M': None if BLIND else exp1,
        'exp2_Angel': None if BLIND else exp2,
        'age': None
    }

    IGNORE_LABELS = {
        'exp1_label_FI_AL_M': ['-1.0', '0.0'],
        'exp2_Angel': ['-1.0', '1.0'],
        'age': None
    }
    return TEST_LABEL_ALIAS[test], IGNORE_LABELS[test]


def behavior_analysis_pipeline(ROOT, dataRoot, test, behavior='both', dim_models='all', ND=3, LD_dim=3,
                              NAN_POLICY='drop', BLIND=False, normalize_mode='true', JNOTEBOOK_MODE=True):
    # TODO: GET RAW MODEL
    """
    :param ROOT:
    :param dataRoot:
    :param behavior:
    :param dim_models: COULD BE 'all' or A SUBLIST OF ['PCA', 'ICA', 'ISOMAP', 'Kernel_PCA']
    :param NAN_POLICY:
    :param JNOTEBOOK_MODE:
    :return:
    """
    # SPECIFICATION
    dataset_name = '4CR'  # 'MAS'
    dataOut = os.path.join(dataRoot, dataset_name)
    plots = os.path.join(ROOT, 'plots', dataset_name)

    if not os.path.exists(plots):
        os.makedirs(plots)

    if not os.path.exists(dataOut):
        os.makedirs(dataOut)
    pdf = load_session(dataRoot, dataset_name)

    # preprocessing
    RANGES = {'FIP': 14,  # Behavior only
              'MAS': list(range(6)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              'MAS2': list(range(8)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              '4CR': 8}  # fix for all
    vectorize_func = {
        'FIP': FIP_data_vectorizer,
        'MAS': MAS_data_vectorizer,
        'MAS2': MAS_data_vectorizer,
        '4CR': FOURCR_data_vectorizer
    }

    test_label_alias, ignore_labels = get_test_specific_options(test, BLIND=BLIND)


    # FEATURE CODING
    # todo: set by slice problem did not show up with step by step run
    (tPDF, tLabels), (xPDF, xLabels) = vectorize_func[dataset_name](pdf, RANGES[dataset_name], NAN_POLICY)
    tLEN, xLEN = tPDF.shape[1], xPDF.shape[1]
    T_nonan_dm, keywordsT, X_nonan_dm, keywordsX = feature_preprocess(tPDF, xPDF)

    # DATA SELECTION
    feature_classes_X = dataset_split_feature_classes(xPDF)
    if behavior == 'SD':
        SDX = feature_classes_X['SD']
        _, _, SDX_nonan_dm, SDkeywordsX = feature_preprocess(tPDF, SDX)
        BXpdf = SDX
        BX_nonan_dm, BkeywordsX = SDX_nonan_dm, SDkeywordsX
    elif behavior == 'REV':
        REVX = feature_classes_X['REV']
        _, _, REVX_nonan_dm, REVkeywordsX = feature_preprocess(tPDF, REVX)
        BXpdf = REVX
        BX_nonan_dm, BkeywordsX = REVX_nonan_dm, REVkeywordsX
    else:
        BXpdf = xPDF
        BX_nonan_dm, BkeywordsX = X_nonan_dm, keywordsX

    STRIFY = True
    if test[:3] == 'exp':
        LABEL = test  # 'exp1_label_FI_AL_M', 'exp2_Angel'
          # Set this as true if we want the color of the points to be discrete (for discrete classes of vars)
        LDLabels = get_data_label(tPDF, tLabels, LABEL, STRIFY=STRIFY)
        BTpdf = tPDF
    elif test == 'age':
        exp1_labels = get_data_label(tPDF, tLabels, 'exp1_label_FI_AL_M', STRIFY=False)
        angel_labels = get_data_label(tPDF, tLabels, 'exp2_Angel', STRIFY=False)
        LABEL = 'Age At Test'
        LDLabels = get_data_label(tPDF, tLabels, LABEL, STRIFY=False)
        # FILTER OUT Treatment Groups
        selectors = (angel_labels.astype(np.float) == -1) & (exp1_labels.astype(np.float) == -1)
        BX_nonan_dm = BX_nonan_dm[selectors]
        LDLabels = LDLabels[selectors]
        BTpdf = tPDF[selectors]
    else:
        raise NotImplementedError(f'Unknown test: {test}')

    # DIM Reduction
    pca_full, ClusteringDim = dataset_dimensionality_probing(BX_nonan_dm, dataset_name,
                                                             JNOTEBOOK_MODE=JNOTEBOOK_MODE)
    models_behaviors = dim_reduction(BX_nonan_dm, ClusteringDim, models=dim_models,
                                     kernelPCA_params=(2, 'poly'))
    visualize_LD_multimodels(models_behaviors, LDLabels, 0, ND=ND, show=True,
                             label_alias=test_label_alias, JNOTEBOOK_MODE=JNOTEBOOK_MODE)
    # Quantify Difference
    # Train Classifiers on X_LDs and quantify cross validation area
    if test == 'age':
        print("regression methods comes soon")
    else:
        clfs, confs = classifier_LD_multimodels(models_behaviors, LDLabels, LD_dim=LD_dim, mode=normalize_mode,
                                                        ignore_labels=ignore_labels)

    return models_behaviors, LDLabels, BXpdf, BTpdf # TODO: Returns tLabels also depending on utility



""" ##########################################
#################### MAIN ####################
########################################## """


def main():
    # TODO: add function to handle merged datasets
    # paths
    JNOTEBOOK_MODE = False
    CAT_ENCODE = None
    NAN_POLICY = 'drop'
    DIM_MODELS = ['ISOMAP']
    TEST_OPTIONS = ['exp1_label_FI_AL_M', 'exp2_Angel', 'age']
    BEHAVIOR_OPTS = ['both', 'SD', 'REV']
    if JNOTEBOOK_MODE:
        ROOT = "/content/drive/My Drive/WilbrechtLab/adversity_4CR/"
    else:
        ROOT = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/PCA_adversity/"
    dataRoot = os.path.join(ROOT, 'data')

    test_p, behavior_p, plot_dimension_p = 'exp2_Angel', 'both', 3
    models_behaviors, LDLabels, BXpdf, BTpdf = behavior_analysis_pipeline(ROOT, dataRoot, test_p,
                                                            behavior=behavior_p, dim_models=DIM_MODELS,
                               ND=plot_dimension_p, LD_dim=3, NAN_POLICY=NAN_POLICY, JNOTEBOOK_MODE=True)

    dataset_name = '4CR' # 'MAS'
    dataOut = os.path.join(dataRoot, dataset_name)
    plots = os.path.join(ROOT, 'plots', dataset_name)

    if not os.path.exists(plots):
        os.makedirs(plots)

    if not os.path.exists(dataOut):
        os.makedirs(dataOut)
    pdf = load_session(dataRoot, dataset_name)

    # preprocessing
    RANGES = {'FIP': 14,  # Behavior only
              'MAS': list(range(6)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              'MAS2': list(range(8)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              '4CR': 8}  # fix for all
    vectorize_func = {
        'FIP': FIP_data_vectorizer,
        'MAS': MAS_data_vectorizer,
        'MAS2': MAS_data_vectorizer,
        '4CR': FOURCR_data_vectorizer
    }

    # todo: set by slice problem did not show up with step by step run
    (tPDF, tLabels), (xPDF, xLabels) = vectorize_func[dataset_name](pdf, RANGES[dataset_name], NAN_POLICY)
    tLEN, xLEN = tPDF.shape[1], xPDF.shape[1]
    T_nonan_dm, keywordsT, X_nonan_dm, keywordsX = feature_preprocess(tPDF, xPDF)
    pca_full, ClusteringDim = dataset_dimensionality_probing(X_nonan_dm, dataset_name,
                                                             JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # Correlation Matrix
    get_corr_matrix(T_nonan_dm, keywordsT, X_nonan_dm, keywordsX, dataset_name,
                    dataOut, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # DATA labeling
    # MAS: ['Genotype', 'Treat']
    # df['Name'] = df['First'].str.cat(df['Last'],sep=" ")
    LABEL = 'Age At Test'  # 'exp1_label_FI_AL_M', 'exp2_Angel'
    STRIFY = True
    LDLabels=get_data_label(tPDF, tLabels, LABEL, STRIFY=STRIFY)

    # DIM Reduction
    models = dim_reduction(X_nonan_dm, ClusteringDim, kernelPCA_params=(2, 'poly'))
    visualize_LD_multimodels(models, LDLabels, 0, ND=3, show=True, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # ------------------------------- LOOP -------------------------------
    # TODO: fix loading methods
    for l in tLabels.columns:
        LDLabels = tLabels[l]
        # PCA
        save_loading_plots(models['PCA'][0], keywordsX, 'PCA', plots)
        save_LD_plots(models['PCA'][1], LDLabels, 'PCA', plots, show=False)

        # ICA
        fastICA = FastICA(n_components=ClusteringDim, random_state=RAND_STATE)
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









