import nibabel as nib
import numpy as np
from numpy import array
from nilearn import plotting
from nilearn.image import concat_imgs, mean_img
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show
from nistats.thresholding import map_threshold
import pandas as pd
import os
import os.path
from os.path import join


dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
# Performing the GLM analysis
# ---------------------------
#
# It is now time to create and estimate a ``FirstLevelModel`` object, that will generate the *design matrix* using the  information provided by the ``events`` object.
###############################################################################

from nistats.first_level_model import FirstLevelModel

###############################################################################
# Formally, we have taken the first design matrix, because the model is
# implictily meant to for multiple runs.
###############################################################################

from nistats.reporting import plot_design_matrix
import matplotlib.pyplot as plt
import os
from os.path import join




def prepRealDataset(image_path):
    """ Prepare a real, existing dataset for use with the simulator

    Read in the supplied 4d image file, set orientation to RAS+

    Parameters
    ----------
    image_path : string
        full path to the dataset you want to use

    Returns
    -------
    ds_RAS : nibabel-like image
        Nibabel dataset with orientation set to RAS+

    """
    print('Prepping dataset: {}'.format(image_path))
    ds = nib.load(image_path)

    # make sure it's RAS+
    ds_RAS = nib.as_closest_canonical(ds)

    print('Dimensions: {}'.format(ds_RAS.shape))
    return ds_RAS


def GLM_Analysis(inputfile, eventfile, TR, trial_type):

    #dataset = prepRealDataset(inputfile)
    dataset = concat_imgs(inputfile)
    mean_dataset = mean_img(dataset)


    ###############################################################################
    # Specifying the experimental paradigm
    ###############################################################################
    events = pd.read_table(eventfile)
    map_trial_type={}
    for items in events['trial_type'].tolist():
        map_trial_type[items] = items

    ###############################################################################
    # Parameters of the first-level model
    #
    # * t_r=1.6(s) is the time of repetition of acquisitions
    # * noise_model='ar1' specifies the noise covariance model: a lag-1 dependence
    # * standardize=False means that we do not want to rescale the time series to mean 0, variance 1
    # * hrf_model='spm' means that we rely on the SPM "canonical hrf" model (without time or dispersion derivatives)
    # * drift_model='cosine' means that we model the signal drifts as slow oscillating time functions
    # * high_pass=0.01(Hz) defines the cutoff frequency (inverse of the time period).
    fmri_glm = FirstLevelModel(t_r=TR,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           high_pass=.01)
    ###############################################################################
    # Now that we have specified the model, we can run it on the fMRI image
    fmri_glm = fmri_glm.fit(dataset, events)

    ###############################################################################
    # One can inspect the design matrix (rows represent time, and
    # columns contain the predictors).
    design_matrix = fmri_glm.design_matrices_[0]
    map_types={}
    for items in design_matrix.dtypes.iteritems(): 
        iteam1,iteam2 = items
        map_types[iteam1]=iteam2
    ###############################################################################
    # Formally, we have taken the first design matrix, because the model is
    # implictily meant to for multiple runs.
    plot_design_matrix(design_matrix)
    #plt.show()
    plt.savefig("designed_matrix1.png")
    ###############################################################################
    # Save the design matrix image to disk
    # first create a directory where you want to write the images

    outdir = 'results'
    if not os.path.exists(outdir):
       os.mkdir(outdir)

    plot_design_matrix(design_matrix, output_file=join(outdir, 'design_matrix.png'))

    ###############################################################################
    # Detecting voxels with significant effects
    # -----------------------------------------
    #
    # To access the estimated coefficients (Betas of the GLM model), we
    # created contrast with a single '1' in each of the columns: The role
    # of the contrast is to select some columns of the model --and
    # potentially weight them-- to study the associated statistics. So in
    # a nutshell, a contrast is a weighted combination of the estimated
    # effects.  Here we can define canonical contrasts that just consider
    # the two condition in isolation ---let's call them "conditions"---
    # then a contrast that makes the difference between these conditions.

    conditions={}

    active_minus_rest = None

    tmp_effects_of_interest = []

    for x in map_trial_type:

        str_arr=''

        for y in map_types.keys():

            if str_arr=='':

               if y==x:

                  str_arr='1.'

               else:

                  str_arr='0.'
            else:

               if y==x:

                  str_arr+=',1.'

               else:

                  str_arr+=',0.' 

        str_arr='['+str_arr+']'

        conditions[x]=array(eval(str_arr))

        tmp_effects_of_interest.append(conditions[x])

        if active_minus_rest is None:

           if trial_type==x:

              active_minus_rest=conditions[x]

           else:

              active_minus_rest=-conditions[x]

        else:

           if trial_type==x:

              active_minus_rest=active_minus_rest+conditions[x]

           else:

              active_minus_rest=active_minus_rest-conditions[x]

    effects_of_interest = np.array(tmp_effects_of_interest)

    ###############################################################################
    # Let's look at it: plot the coefficients of the contrast, indexed by
    # the names of the columns of the design matrix.

    from nistats.reporting import plot_contrast_matrix
    plot_contrast_matrix(active_minus_rest, design_matrix=design_matrix)

    ###############################################################################
    # Below, we compute the estimated effect. It is in BOLD signal unit,
    # but has no statistical guarantees, because it does not take into
    # account the associated variance.
    eff_map = fmri_glm.compute_contrast(active_minus_rest,output_type='effect_size')
    ###############################################################################
    # In order to get statistical significance, we form a t-statistic, and
    # directly convert is into z-scale. The z-scale means that the values
    # are scaled to match a standard Gaussian distribution (mean=0,
    # variance=1), across voxels, if there were now effects in the data.

    z_map = fmri_glm.compute_contrast(active_minus_rest,output_type='z_score')

    ###############################################################################
    # Plot thresholded z scores map.
    #
    # We display it on top of the average
    # functional image of the series (could be the anatomical image of the
    # subject).  We use arbitrarily a threshold of 3.0 in z-scale. We'll
    # see later how to use corrected thresholds. We will show 3
    # axial views, with display_mode='z' and cut_coords=3

    ###############################################################################
    # Statistical significance testing. One should worry about the
    # statistical validity of the procedure: here we used an arbitrary
    # threshold of 3.0 but the threshold should provide some guarantees on
    # the risk of false detections (aka type-1 errors in statistics).
    # One suggestion is to control the false positive rate (fpr, denoted by
    # alpha) at a certain level, e.g. 0.001: this means that there is 0.1% chance
    # of declaring an inactive voxel, active.

    from nistats.thresholding import map_threshold
    _, threshold = map_threshold(z_map, alpha=.001, height_control='fpr')
    ###############################################################################
    # The problem is that with this you expect 0.001 * n_voxels to show up
    # while they're not active --- tens to hundreds of voxels. A more
    # conservative solution is to control the family wise error rate,
    # i.e. the probability of making only one false detection, say at
    # 5%. For that we use the so-called Bonferroni correction

    _, threshold = map_threshold(z_map, alpha=.05, height_control='bonferroni')
    ###############################################################################
    # This is quite conservative indeed!  A popular alternative is to
    # control the expected proportion of
    # false discoveries among detections. This is called the false
    # discovery rate

    _, threshold = map_threshold(z_map, alpha=.05, height_control='fdr')
    ###############################################################################
    # Finally people like to discard isolated voxels (aka "small
    # clusters") from these images. It is possible to generate a
    # thresholded map with small clusters removed by providing a
    # cluster_threshold argument. Here clusters smaller than 10 voxels
    # will be discarded.

    clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=10)


    ###############################################################################
    # We can save the effect and zscore maps to the disk
    z_map.to_filename(join(outdir, 'active_vs_rest_z_map1.nii.gz'))
    eff_map.to_filename(join(outdir, 'active_vs_rest_eff_map1.nii.gz'))

    ###############################################################################

    ###############################################################################
    # Performing an F-test
    #
    # "active vs rest" is a typical t test: condition versus
    # baseline. Another popular type of test is an F test in which one
    # seeks whether a certain combination of conditions (possibly two-,
    # three- or higher-dimensional) explains a significant proportion of
    # the signal.  Here one might for instance test which voxels are well
    # explained by combination of the active and rest condition.
    plot_contrast_matrix(effects_of_interest, design_matrix)
    plt.savefig("designed_matrix2.png")
    ###############################################################################
    # Specify the contrast and compute the corresponding map. Actually, the
    # contrast specification is done exactly the same way as for t-
    # contrasts.

    z_map = fmri_glm.compute_contrast(effects_of_interest,output_type='z_score')
    ###############################################################################
    # Note that the statistic has been converted to a z-variable, which
    # makes it easier to represent it.

    clean_map, threshold = map_threshold(z_map, alpha=.05, height_control='fdr', cluster_threshold=5)
    plot_stat_map(clean_map, bg_img=mean_dataset, threshold=threshold,display_mode='z', cut_coords=3, black_bg=True,title='Effects of interest (fdr=0.05), clusters > 5 voxels')
    plt.show()
    plt.savefig("active_map.png")
    plt.close('all')
    ###############################################################################
    # Oops, there is a lot of non-neural signal in there (ventricles, arteries)...
#inputfile='/home/pritom/nistats/data/sub04/func/sub-04_task-passiveimageviewing_bold.nii'

inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii']

inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii','/home/pritom/nistates/update/temp/func/11/passiveimageviewing_bold_11.nii','/home/pritom/nistates/update/temp/func/12/passiveimageviewing_bold_12.nii','/home/pritom/nistates/update/temp/func/13/passiveimageviewing_bold_13.nii','/home/pritom/nistates/update/temp/func/14/passiveimageviewing_bold_14.nii','/home/pritom/nistates/update/temp/func/15/passiveimageviewing_bold_15.nii','/home/pritom/nistates/update/temp/func/16/passiveimageviewing_bold_16.nii','/home/pritom/nistates/update/temp/func/17/passiveimageviewing_bold_17.nii','/home/pritom/nistates/update/temp/func/18/passiveimageviewing_bold_18.nii','/home/pritom/nistates/update/temp/func/19/passiveimageviewing_bold_19.nii','/home/pritom/nistates/update/temp/func/20/passiveimageviewing_bold_20.nii']



inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii','/home/pritom/nistates/update/temp/func/11/passiveimageviewing_bold_11.nii','/home/pritom/nistates/update/temp/func/12/passiveimageviewing_bold_12.nii','/home/pritom/nistates/update/temp/func/13/passiveimageviewing_bold_13.nii','/home/pritom/nistates/update/temp/func/14/passiveimageviewing_bold_14.nii','/home/pritom/nistates/update/temp/func/15/passiveimageviewing_bold_15.nii','/home/pritom/nistates/update/temp/func/16/passiveimageviewing_bold_16.nii','/home/pritom/nistates/update/temp/func/17/passiveimageviewing_bold_17.nii','/home/pritom/nistates/update/temp/func/18/passiveimageviewing_bold_18.nii','/home/pritom/nistates/update/temp/func/19/passiveimageviewing_bold_19.nii','/home/pritom/nistates/update/temp/func/20/passiveimageviewing_bold_20.nii','/home/pritom/nistates/update/temp/func/21/passiveimageviewing_bold_21.nii','/home/pritom/nistates/update/temp/func/22/passiveimageviewing_bold_22.nii','/home/pritom/nistates/update/temp/func/23/passiveimageviewing_bold_23.nii','/home/pritom/nistates/update/temp/func/24/passiveimageviewing_bold_24.nii','/home/pritom/nistates/update/temp/func/25/passiveimageviewing_bold_25.nii','/home/pritom/nistates/update/temp/func/26/passiveimageviewing_bold_26.nii','/home/pritom/nistates/update/temp/func/27/passiveimageviewing_bold_27.nii','/home/pritom/nistates/update/temp/func/28/passiveimageviewing_bold_28.nii','/home/pritom/nistates/update/temp/func/29/passiveimageviewing_bold_29.nii','/home/pritom/nistates/update/temp/func/30/passiveimageviewing_bold_30.nii']

inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii','/home/pritom/nistates/update/temp/func/11/passiveimageviewing_bold_11.nii','/home/pritom/nistates/update/temp/func/12/passiveimageviewing_bold_12.nii','/home/pritom/nistates/update/temp/func/13/passiveimageviewing_bold_13.nii','/home/pritom/nistates/update/temp/func/14/passiveimageviewing_bold_14.nii','/home/pritom/nistates/update/temp/func/15/passiveimageviewing_bold_15.nii','/home/pritom/nistates/update/temp/func/16/passiveimageviewing_bold_16.nii','/home/pritom/nistates/update/temp/func/17/passiveimageviewing_bold_17.nii','/home/pritom/nistates/update/temp/func/18/passiveimageviewing_bold_18.nii','/home/pritom/nistates/update/temp/func/19/passiveimageviewing_bold_19.nii','/home/pritom/nistates/update/temp/func/20/passiveimageviewing_bold_20.nii','/home/pritom/nistates/update/temp/func/21/passiveimageviewing_bold_21.nii','/home/pritom/nistates/update/temp/func/22/passiveimageviewing_bold_22.nii','/home/pritom/nistates/update/temp/func/23/passiveimageviewing_bold_23.nii','/home/pritom/nistates/update/temp/func/24/passiveimageviewing_bold_24.nii','/home/pritom/nistates/update/temp/func/25/passiveimageviewing_bold_25.nii','/home/pritom/nistates/update/temp/func/26/passiveimageviewing_bold_26.nii','/home/pritom/nistates/update/temp/func/27/passiveimageviewing_bold_27.nii','/home/pritom/nistates/update/temp/func/28/passiveimageviewing_bold_28.nii','/home/pritom/nistates/update/temp/func/29/passiveimageviewing_bold_29.nii','/home/pritom/nistates/update/temp/func/30/passiveimageviewing_bold_30.nii','/home/pritom/nistates/update/temp/func/31/passiveimageviewing_bold_31.nii','/home/pritom/nistates/update/temp/func/32/passiveimageviewing_bold_32.nii','/home/pritom/nistates/update/temp/func/33/passiveimageviewing_bold_33.nii','/home/pritom/nistates/update/temp/func/34/passiveimageviewing_bold_34.nii','/home/pritom/nistates/update/temp/func/35/passiveimageviewing_bold_35.nii','/home/pritom/nistates/update/temp/func/36/passiveimageviewing_bold_36.nii','/home/pritom/nistates/update/temp/func/37/passiveimageviewing_bold_37.nii','/home/pritom/nistates/update/temp/func/38/passiveimageviewing_bold_38.nii','/home/pritom/nistates/update/temp/func/39/passiveimageviewing_bold_39.nii','/home/pritom/nistates/update/temp/func/40/passiveimageviewing_bold_40.nii']

inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii','/home/pritom/nistates/update/temp/func/11/passiveimageviewing_bold_11.nii','/home/pritom/nistates/update/temp/func/12/passiveimageviewing_bold_12.nii','/home/pritom/nistates/update/temp/func/13/passiveimageviewing_bold_13.nii','/home/pritom/nistates/update/temp/func/14/passiveimageviewing_bold_14.nii','/home/pritom/nistates/update/temp/func/15/passiveimageviewing_bold_15.nii','/home/pritom/nistates/update/temp/func/16/passiveimageviewing_bold_16.nii','/home/pritom/nistates/update/temp/func/17/passiveimageviewing_bold_17.nii','/home/pritom/nistates/update/temp/func/18/passiveimageviewing_bold_18.nii','/home/pritom/nistates/update/temp/func/19/passiveimageviewing_bold_19.nii','/home/pritom/nistates/update/temp/func/20/passiveimageviewing_bold_20.nii','/home/pritom/nistates/update/temp/func/21/passiveimageviewing_bold_21.nii','/home/pritom/nistates/update/temp/func/22/passiveimageviewing_bold_22.nii','/home/pritom/nistates/update/temp/func/23/passiveimageviewing_bold_23.nii','/home/pritom/nistates/update/temp/func/24/passiveimageviewing_bold_24.nii','/home/pritom/nistates/update/temp/func/25/passiveimageviewing_bold_25.nii','/home/pritom/nistates/update/temp/func/26/passiveimageviewing_bold_26.nii','/home/pritom/nistates/update/temp/func/27/passiveimageviewing_bold_27.nii','/home/pritom/nistates/update/temp/func/28/passiveimageviewing_bold_28.nii','/home/pritom/nistates/update/temp/func/29/passiveimageviewing_bold_29.nii','/home/pritom/nistates/update/temp/func/30/passiveimageviewing_bold_30.nii','/home/pritom/nistates/update/temp/func/31/passiveimageviewing_bold_31.nii','/home/pritom/nistates/update/temp/func/32/passiveimageviewing_bold_32.nii','/home/pritom/nistates/update/temp/func/33/passiveimageviewing_bold_33.nii','/home/pritom/nistates/update/temp/func/34/passiveimageviewing_bold_34.nii','/home/pritom/nistates/update/temp/func/35/passiveimageviewing_bold_35.nii','/home/pritom/nistates/update/temp/func/36/passiveimageviewing_bold_36.nii','/home/pritom/nistates/update/temp/func/37/passiveimageviewing_bold_37.nii','/home/pritom/nistates/update/temp/func/38/passiveimageviewing_bold_38.nii','/home/pritom/nistates/update/temp/func/39/passiveimageviewing_bold_39.nii','/home/pritom/nistates/update/temp/func/40/passiveimageviewing_bold_40.nii','/home/pritom/nistates/update/temp/func/41/passiveimageviewing_bold_41.nii','/home/pritom/nistates/update/temp/func/42/passiveimageviewing_bold_42.nii','/home/pritom/nistates/update/temp/func/43/passiveimageviewing_bold_43.nii','/home/pritom/nistates/update/temp/func/44/passiveimageviewing_bold_44.nii','/home/pritom/nistates/update/temp/func/45/passiveimageviewing_bold_45.nii','/home/pritom/nistates/update/temp/func/46/passiveimageviewing_bold_46.nii','/home/pritom/nistates/update/temp/func/47/passiveimageviewing_bold_47.nii','/home/pritom/nistates/update/temp/func/48/passiveimageviewing_bold_48.nii','/home/pritom/nistates/update/temp/func/49/passiveimageviewing_bold_49.nii','/home/pritom/nistates/update/temp/func/50/passiveimageviewing_bold_50.nii']

inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii','/home/pritom/nistates/update/temp/func/11/passiveimageviewing_bold_11.nii','/home/pritom/nistates/update/temp/func/12/passiveimageviewing_bold_12.nii','/home/pritom/nistates/update/temp/func/13/passiveimageviewing_bold_13.nii','/home/pritom/nistates/update/temp/func/14/passiveimageviewing_bold_14.nii','/home/pritom/nistates/update/temp/func/15/passiveimageviewing_bold_15.nii','/home/pritom/nistates/update/temp/func/16/passiveimageviewing_bold_16.nii','/home/pritom/nistates/update/temp/func/17/passiveimageviewing_bold_17.nii','/home/pritom/nistates/update/temp/func/18/passiveimageviewing_bold_18.nii','/home/pritom/nistates/update/temp/func/19/passiveimageviewing_bold_19.nii','/home/pritom/nistates/update/temp/func/20/passiveimageviewing_bold_20.nii','/home/pritom/nistates/update/temp/func/21/passiveimageviewing_bold_21.nii','/home/pritom/nistates/update/temp/func/22/passiveimageviewing_bold_22.nii','/home/pritom/nistates/update/temp/func/23/passiveimageviewing_bold_23.nii','/home/pritom/nistates/update/temp/func/24/passiveimageviewing_bold_24.nii','/home/pritom/nistates/update/temp/func/25/passiveimageviewing_bold_25.nii','/home/pritom/nistates/update/temp/func/26/passiveimageviewing_bold_26.nii','/home/pritom/nistates/update/temp/func/27/passiveimageviewing_bold_27.nii','/home/pritom/nistates/update/temp/func/28/passiveimageviewing_bold_28.nii','/home/pritom/nistates/update/temp/func/29/passiveimageviewing_bold_29.nii','/home/pritom/nistates/update/temp/func/30/passiveimageviewing_bold_30.nii','/home/pritom/nistates/update/temp/func/31/passiveimageviewing_bold_31.nii','/home/pritom/nistates/update/temp/func/32/passiveimageviewing_bold_32.nii','/home/pritom/nistates/update/temp/func/33/passiveimageviewing_bold_33.nii','/home/pritom/nistates/update/temp/func/34/passiveimageviewing_bold_34.nii','/home/pritom/nistates/update/temp/func/35/passiveimageviewing_bold_35.nii','/home/pritom/nistates/update/temp/func/36/passiveimageviewing_bold_36.nii','/home/pritom/nistates/update/temp/func/37/passiveimageviewing_bold_37.nii','/home/pritom/nistates/update/temp/func/38/passiveimageviewing_bold_38.nii','/home/pritom/nistates/update/temp/func/39/passiveimageviewing_bold_39.nii','/home/pritom/nistates/update/temp/func/40/passiveimageviewing_bold_40.nii','/home/pritom/nistates/update/temp/func/41/passiveimageviewing_bold_41.nii','/home/pritom/nistates/update/temp/func/42/passiveimageviewing_bold_42.nii','/home/pritom/nistates/update/temp/func/43/passiveimageviewing_bold_43.nii','/home/pritom/nistates/update/temp/func/44/passiveimageviewing_bold_44.nii','/home/pritom/nistates/update/temp/func/45/passiveimageviewing_bold_45.nii','/home/pritom/nistates/update/temp/func/46/passiveimageviewing_bold_46.nii','/home/pritom/nistates/update/temp/func/47/passiveimageviewing_bold_47.nii','/home/pritom/nistates/update/temp/func/48/passiveimageviewing_bold_48.nii','/home/pritom/nistates/update/temp/func/49/passiveimageviewing_bold_49.nii','/home/pritom/nistates/update/temp/func/50/passiveimageviewing_bold_50.nii','/home/pritom/nistates/update/temp/func/51/passiveimageviewing_bold_51.nii','/home/pritom/nistates/update/temp/func/52/passiveimageviewing_bold_52.nii','/home/pritom/nistates/update/temp/func/53/passiveimageviewing_bold_53.nii','/home/pritom/nistates/update/temp/func/54/passiveimageviewing_bold_54.nii','/home/pritom/nistates/update/temp/func/55/passiveimageviewing_bold_55.nii','/home/pritom/nistates/update/temp/func/56/passiveimageviewing_bold_56.nii','/home/pritom/nistates/update/temp/func/57/passiveimageviewing_bold_57.nii','/home/pritom/nistates/update/temp/func/58/passiveimageviewing_bold_58.nii','/home/pritom/nistates/update/temp/func/59/passiveimageviewing_bold_59.nii','/home/pritom/nistates/update/temp/func/60/passiveimageviewing_bold_60.nii']

inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii','/home/pritom/nistates/update/temp/func/11/passiveimageviewing_bold_11.nii','/home/pritom/nistates/update/temp/func/12/passiveimageviewing_bold_12.nii','/home/pritom/nistates/update/temp/func/13/passiveimageviewing_bold_13.nii','/home/pritom/nistates/update/temp/func/14/passiveimageviewing_bold_14.nii','/home/pritom/nistates/update/temp/func/15/passiveimageviewing_bold_15.nii','/home/pritom/nistates/update/temp/func/16/passiveimageviewing_bold_16.nii','/home/pritom/nistates/update/temp/func/17/passiveimageviewing_bold_17.nii','/home/pritom/nistates/update/temp/func/18/passiveimageviewing_bold_18.nii','/home/pritom/nistates/update/temp/func/19/passiveimageviewing_bold_19.nii','/home/pritom/nistates/update/temp/func/20/passiveimageviewing_bold_20.nii','/home/pritom/nistates/update/temp/func/21/passiveimageviewing_bold_21.nii','/home/pritom/nistates/update/temp/func/22/passiveimageviewing_bold_22.nii','/home/pritom/nistates/update/temp/func/23/passiveimageviewing_bold_23.nii','/home/pritom/nistates/update/temp/func/24/passiveimageviewing_bold_24.nii','/home/pritom/nistates/update/temp/func/25/passiveimageviewing_bold_25.nii','/home/pritom/nistates/update/temp/func/26/passiveimageviewing_bold_26.nii','/home/pritom/nistates/update/temp/func/27/passiveimageviewing_bold_27.nii','/home/pritom/nistates/update/temp/func/28/passiveimageviewing_bold_28.nii','/home/pritom/nistates/update/temp/func/29/passiveimageviewing_bold_29.nii','/home/pritom/nistates/update/temp/func/30/passiveimageviewing_bold_30.nii','/home/pritom/nistates/update/temp/func/31/passiveimageviewing_bold_31.nii','/home/pritom/nistates/update/temp/func/32/passiveimageviewing_bold_32.nii','/home/pritom/nistates/update/temp/func/33/passiveimageviewing_bold_33.nii','/home/pritom/nistates/update/temp/func/34/passiveimageviewing_bold_34.nii','/home/pritom/nistates/update/temp/func/35/passiveimageviewing_bold_35.nii','/home/pritom/nistates/update/temp/func/36/passiveimageviewing_bold_36.nii','/home/pritom/nistates/update/temp/func/37/passiveimageviewing_bold_37.nii','/home/pritom/nistates/update/temp/func/38/passiveimageviewing_bold_38.nii','/home/pritom/nistates/update/temp/func/39/passiveimageviewing_bold_39.nii','/home/pritom/nistates/update/temp/func/40/passiveimageviewing_bold_40.nii','/home/pritom/nistates/update/temp/func/41/passiveimageviewing_bold_41.nii','/home/pritom/nistates/update/temp/func/42/passiveimageviewing_bold_42.nii','/home/pritom/nistates/update/temp/func/43/passiveimageviewing_bold_43.nii','/home/pritom/nistates/update/temp/func/44/passiveimageviewing_bold_44.nii','/home/pritom/nistates/update/temp/func/45/passiveimageviewing_bold_45.nii','/home/pritom/nistates/update/temp/func/46/passiveimageviewing_bold_46.nii','/home/pritom/nistates/update/temp/func/47/passiveimageviewing_bold_47.nii','/home/pritom/nistates/update/temp/func/48/passiveimageviewing_bold_48.nii','/home/pritom/nistates/update/temp/func/49/passiveimageviewing_bold_49.nii','/home/pritom/nistates/update/temp/func/50/passiveimageviewing_bold_50.nii','/home/pritom/nistates/update/temp/func/51/passiveimageviewing_bold_51.nii','/home/pritom/nistates/update/temp/func/52/passiveimageviewing_bold_52.nii','/home/pritom/nistates/update/temp/func/53/passiveimageviewing_bold_53.nii','/home/pritom/nistates/update/temp/func/54/passiveimageviewing_bold_54.nii','/home/pritom/nistates/update/temp/func/55/passiveimageviewing_bold_55.nii','/home/pritom/nistates/update/temp/func/56/passiveimageviewing_bold_56.nii','/home/pritom/nistates/update/temp/func/57/passiveimageviewing_bold_57.nii','/home/pritom/nistates/update/temp/func/58/passiveimageviewing_bold_58.nii','/home/pritom/nistates/update/temp/func/59/passiveimageviewing_bold_59.nii','/home/pritom/nistates/update/temp/func/60/passiveimageviewing_bold_60.nii','/home/pritom/nistates/update/temp/func/61/passiveimageviewing_bold_61.nii','/home/pritom/nistates/update/temp/func/62/passiveimageviewing_bold_62.nii','/home/pritom/nistates/update/temp/func/63/passiveimageviewing_bold_63.nii','/home/pritom/nistates/update/temp/func/64/passiveimageviewing_bold_64.nii','/home/pritom/nistates/update/temp/func/65/passiveimageviewing_bold_65.nii','/home/pritom/nistates/update/temp/func/66/passiveimageviewing_bold_66.nii','/home/pritom/nistates/update/temp/func/67/passiveimageviewing_bold_67.nii','/home/pritom/nistates/update/temp/func/68/passiveimageviewing_bold_68.nii','/home/pritom/nistates/update/temp/func/69/passiveimageviewing_bold_69.nii','/home/pritom/nistates/update/temp/func/70/passiveimageviewing_bold_70.nii']

inputfile =['/home/pritom/nistates/update/temp/func/1/passiveimageviewing_bold_1.nii','/home/pritom/nistates/update/temp/func/2/passiveimageviewing_bold_2.nii','/home/pritom/nistates/update/temp/func/3/passiveimageviewing_bold_3.nii','/home/pritom/nistates/update/temp/func/4/passiveimageviewing_bold_4.nii','/home/pritom/nistates/update/temp/func/5/passiveimageviewing_bold_5.nii','/home/pritom/nistates/update/temp/func/6/passiveimageviewing_bold_6.nii','/home/pritom/nistates/update/temp/func/7/passiveimageviewing_bold_7.nii','/home/pritom/nistates/update/temp/func/8/passiveimageviewing_bold_8.nii','/home/pritom/nistates/update/temp/func/9/passiveimageviewing_bold_9.nii','/home/pritom/nistates/update/temp/func/10/passiveimageviewing_bold_10.nii','/home/pritom/nistates/update/temp/func/11/passiveimageviewing_bold_11.nii','/home/pritom/nistates/update/temp/func/12/passiveimageviewing_bold_12.nii','/home/pritom/nistates/update/temp/func/13/passiveimageviewing_bold_13.nii','/home/pritom/nistates/update/temp/func/14/passiveimageviewing_bold_14.nii','/home/pritom/nistates/update/temp/func/15/passiveimageviewing_bold_15.nii','/home/pritom/nistates/update/temp/func/16/passiveimageviewing_bold_16.nii','/home/pritom/nistates/update/temp/func/17/passiveimageviewing_bold_17.nii','/home/pritom/nistates/update/temp/func/18/passiveimageviewing_bold_18.nii','/home/pritom/nistates/update/temp/func/19/passiveimageviewing_bold_19.nii','/home/pritom/nistates/update/temp/func/20/passiveimageviewing_bold_20.nii','/home/pritom/nistates/update/temp/func/21/passiveimageviewing_bold_21.nii','/home/pritom/nistates/update/temp/func/22/passiveimageviewing_bold_22.nii','/home/pritom/nistates/update/temp/func/23/passiveimageviewing_bold_23.nii','/home/pritom/nistates/update/temp/func/24/passiveimageviewing_bold_24.nii','/home/pritom/nistates/update/temp/func/25/passiveimageviewing_bold_25.nii','/home/pritom/nistates/update/temp/func/26/passiveimageviewing_bold_26.nii','/home/pritom/nistates/update/temp/func/27/passiveimageviewing_bold_27.nii','/home/pritom/nistates/update/temp/func/28/passiveimageviewing_bold_28.nii','/home/pritom/nistates/update/temp/func/29/passiveimageviewing_bold_29.nii','/home/pritom/nistates/update/temp/func/30/passiveimageviewing_bold_30.nii','/home/pritom/nistates/update/temp/func/31/passiveimageviewing_bold_31.nii','/home/pritom/nistates/update/temp/func/32/passiveimageviewing_bold_32.nii','/home/pritom/nistates/update/temp/func/33/passiveimageviewing_bold_33.nii','/home/pritom/nistates/update/temp/func/34/passiveimageviewing_bold_34.nii','/home/pritom/nistates/update/temp/func/35/passiveimageviewing_bold_35.nii','/home/pritom/nistates/update/temp/func/36/passiveimageviewing_bold_36.nii','/home/pritom/nistates/update/temp/func/37/passiveimageviewing_bold_37.nii','/home/pritom/nistates/update/temp/func/38/passiveimageviewing_bold_38.nii','/home/pritom/nistates/update/temp/func/39/passiveimageviewing_bold_39.nii','/home/pritom/nistates/update/temp/func/40/passiveimageviewing_bold_40.nii','/home/pritom/nistates/update/temp/func/41/passiveimageviewing_bold_41.nii','/home/pritom/nistates/update/temp/func/42/passiveimageviewing_bold_42.nii','/home/pritom/nistates/update/temp/func/43/passiveimageviewing_bold_43.nii','/home/pritom/nistates/update/temp/func/44/passiveimageviewing_bold_44.nii','/home/pritom/nistates/update/temp/func/45/passiveimageviewing_bold_45.nii','/home/pritom/nistates/update/temp/func/46/passiveimageviewing_bold_46.nii','/home/pritom/nistates/update/temp/func/47/passiveimageviewing_bold_47.nii','/home/pritom/nistates/update/temp/func/48/passiveimageviewing_bold_48.nii','/home/pritom/nistates/update/temp/func/49/passiveimageviewing_bold_49.nii','/home/pritom/nistates/update/temp/func/50/passiveimageviewing_bold_50.nii','/home/pritom/nistates/update/temp/func/51/passiveimageviewing_bold_51.nii','/home/pritom/nistates/update/temp/func/52/passiveimageviewing_bold_52.nii','/home/pritom/nistates/update/temp/func/53/passiveimageviewing_bold_53.nii','/home/pritom/nistates/update/temp/func/54/passiveimageviewing_bold_54.nii','/home/pritom/nistates/update/temp/func/55/passiveimageviewing_bold_55.nii','/home/pritom/nistates/update/temp/func/56/passiveimageviewing_bold_56.nii','/home/pritom/nistates/update/temp/func/57/passiveimageviewing_bold_57.nii','/home/pritom/nistates/update/temp/func/58/passiveimageviewing_bold_58.nii','/home/pritom/nistates/update/temp/func/59/passiveimageviewing_bold_59.nii','/home/pritom/nistates/update/temp/func/60/passiveimageviewing_bold_60.nii','/home/pritom/nistates/update/temp/func/61/passiveimageviewing_bold_61.nii','/home/pritom/nistates/update/temp/func/62/passiveimageviewing_bold_62.nii','/home/pritom/nistates/update/temp/func/63/passiveimageviewing_bold_63.nii','/home/pritom/nistates/update/temp/func/64/passiveimageviewing_bold_64.nii','/home/pritom/nistates/update/temp/func/65/passiveimageviewing_bold_65.nii','/home/pritom/nistates/update/temp/func/66/passiveimageviewing_bold_66.nii','/home/pritom/nistates/update/temp/func/67/passiveimageviewing_bold_67.nii','/home/pritom/nistates/update/temp/func/68/passiveimageviewing_bold_68.nii','/home/pritom/nistates/update/temp/func/69/passiveimageviewing_bold_69.nii','/home/pritom/nistates/update/temp/func/70/passiveimageviewing_bold_70.nii','/home/pritom/nistates/update/temp/func/71/passiveimageviewing_bold_71.nii','/home/pritom/nistates/update/temp/func/72/passiveimageviewing_bold_72.nii','/home/pritom/nistates/update/temp/func/73/passiveimageviewing_bold_73.nii','/home/pritom/nistates/update/temp/func/74/passiveimageviewing_bold_74.nii','/home/pritom/nistates/update/temp/func/75/passiveimageviewing_bold_75.nii','/home/pritom/nistates/update/temp/func/76/passiveimageviewing_bold_76.nii','/home/pritom/nistates/update/temp/func/77/passiveimageviewing_bold_77.nii','/home/pritom/nistates/update/temp/func/78/passiveimageviewing_bold_78.nii','/home/pritom/nistates/update/temp/func/79/passiveimageviewing_bold_79.nii','/home/pritom/nistates/update/temp/func/80/passiveimageviewing_bold_80.nii']


eventfile='/home/pritom/nistates/update/temp/file_event.tsv'
TR=1.6
trial_type='food'
#GLM_Analysis(inputfile, eventfile, TR, trial_type)
def construct_file_list(timesteps):
    volIdx = 1
    input_file_list=[]
    while(True):

         #print(timesteps)

         #print(dir_path)

         path = join(dir_path, 'temp', 'func/'+str(volIdx) )

         file_name = 'passiveimageviewing_bold_'+str(volIdx)+'.nii'

         filename = join(path, file_name)

         if os.path.isfile(filename):

            input_file_list.append(filename)

            #print ("File exist")

            #print(volIdx)

            if len(input_file_list)>=10 and len(input_file_list)%10==0:

              GLM_Analysis(input_file_list, eventfile, TR, trial_type)

            volIdx+=1

         else:

            print ("File not exist")

         if(volIdx==timesteps):

            break

    #print(input_file_list)


construct_file_list(375)
