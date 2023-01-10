import sys
sys.path.append(r'..\feature-importance')

from functions.general import get_feature_groups
from functions.result_functions import update_results, save_final_results
from math import factorial
import numpy as np
import os
from tqdm.auto import tqdm

def subset_probabilities(n_elements):
    """ get the probability of a subset of length n for for a set of length n_elements"""
    def total_possible_sets(n_elements):
        n_elements = int(n_elements)
        return (2**(n_elements))-1

    def possible_sets_length(n_elements, subset_length):
        numerator = factorial(n_elements)
        denominator = factorial(subset_length) * factorial(n_elements-subset_length)
        return numerator//denominator

    probability = []
    n_total = total_possible_sets(n_elements)
    for subset_length in range(1, n_elements+1):
        subsets_possible = possible_sets_length(n_elements, subset_length)
        probability.append(subsets_possible/n_total)
    return probability

def get_imputed_images(x, y, batch_size, feature_list, config, n_runs=None, feature_index=None):
    m_min = config["mode_specific_params"]["SHAP"]["SHAP_min_samples"]
    aggregated_img = np.load(r'explainer\aggregated_image.npy').reshape(-1, 128,128)
    join_images, unjoined_images = np.array([]), np.array([])

    # sample a number of coalitions for every sample in the batch
    possible_coalitions_lengths = np.arange(1, len(feature_list)) # max lenght = len(feature_list)-1 since np.arange(start, end) includes start but excludes end
    p = subset_probabilities(max(possible_coalitions_lengths))
    p = np.arange(len(possible_coalitions_lengths))
    p = p/np.sum(p)
    sampled_lengths = np.random.choice(possible_coalitions_lengths, size=batch_size, p=p)
    if feature_index is None:
        feature_index_provided = False
    else:
        feature_index_provided = True

    for sample in range(batch_size):
        if feature_index_provided == False:
            feature_index = int(n_runs / m_min)
            n_runs += 1
        other_players = np.delete(feature_list, feature_index, axis=0)
        # sample a random image with a random coalition. coalition = include indexes, imputation = exclude indexes
        img_index = np.random.randint(len(x))
        img = x[img_index].copy()

        n_exclude = len(other_players) - sampled_lengths[sample]
        exclude_indexes = np.random.choice(len(other_players), size=n_exclude, replace=False)

        # impute superpixels that are not in the coalition
        for index in exclude_indexes:
            exclude_feature = other_players[index]
            random_img_number = np.random.randint(len(aggregated_img))
            random_img = aggregated_img[random_img_number]
            img[exclude_feature] = random_img[exclude_feature]

        # create the join and not join state. for the not join state we need to impute the relevant feature
        join_state = img.copy()
        unjoined_state = img.copy()

        # impute the relevant feature
        feature_superpixel = feature_list[feature_index]
        random_img = aggregated_img[np.random.randint(len(aggregated_img))]
        random_img_number = np.random.randint(len(aggregated_img))
        random_img = aggregated_img[random_img_number]
        unjoined_state[feature_superpixel] = random_img[feature_superpixel]
        if len(join_images) > 0:
            join_images = np.concatenate((join_images, [join_state]), axis=0)
            unjoined_images = np.concatenate((unjoined_images, [unjoined_state]), axis=0)
            image_labels = np.concatenate((image_labels, [y[img_index]]), axis=0)
            image_feature_nums = np.concatenate((image_feature_nums, [feature_index]), axis=0)
        else:
            join_images = np.array([join_state])
            unjoined_images = np.array([unjoined_state])
            image_labels = np.array([y[img_index]])
            image_feature_nums = np.array([feature_index])
    data = (join_images, unjoined_images, image_labels, image_feature_nums)
    if n_runs is None:
        return data
    else:
        return data, n_runs

def get_contribution_single(x, y, feature_list, feature_index, model, config, measure='accuracy'):
    """ get the contribution of a single feature to the prediction of a model"""
    batch_size = 1
    data = get_imputed_images(x, y, batch_size, feature_list, config, feature_index=feature_index)
    if data[3] != feature_index:
        raise ValueError("feature index is not correct")
    join_images, unjoined_images = data[0], data[1]
    image_labels = data[2]

    # get the predictions & contributions
    pred_join = model.predict(join_images, verbose=0)
    pred_unjoined = model.predict(unjoined_images, verbose=0)

    if measure == "accuracy":
        y_pred_join = np.argmax(pred_join, axis=1)
        y_pred_unjoined = np.argmax(pred_unjoined, axis=1)
        join_correct = np.sum(y_pred_join == image_labels)
        unjoined_correct = np.sum(y_pred_unjoined == image_labels)
        contribution = join_correct - unjoined_correct
    elif measure == "cross_entropy":
        # still needs to be done
        pass

    return contribution


def get_contribution_batch(x, y, model, n_runs, batch_size, feature_list, config):
    data, n_runs = get_imputed_images(x, y, batch_size, feature_list, config, n_runs=n_runs)
    join_images, unjoined_images = data[0], data[1]
    image_labels = data[2]
    image_batch = np.concatenate((join_images, unjoined_images), axis=0)

    # get the predictions & contributions
    pred_arr = model.predict_on_batch(image_batch)
    y_pred = np.argmax(pred_arr, axis=1)
    pred_join_correct = np.where(y_pred == np.repeat(image_labels, 2), 1, 0)

    # get correct / false predictions for join and unjoin
    join_correct = pred_join_correct[:len(join_images)]
    unjoin_correct = pred_join_correct[len(join_images):]
    contribution_list = join_correct - unjoin_correct

    contribution_dict={}
    image_feature_nums = data[3]
    for feature_index, contribution in zip(image_feature_nums, contribution_list):
        if feature_index in contribution_dict.keys():
            contribution_dict[feature_index] = np.concatenate((contribution_dict[feature_index], [contribution]), axis=0)
        else:
            contribution_dict[feature_index] = np.array([contribution])
    return contribution_dict, n_runs

def calculate_var_score(shapley_values_instance):
    n_samples = shapley_values_instance['n_samples']
    contribution_values = shapley_values_instance['values']
    var = np.var(contribution_values)
    var_score = np.sqrt(var/n_samples) - np.sqrt(var/(n_samples+1))
    return var_score

def SHAP(x, y, model, config, results):
    feature_list = get_feature_groups(config)
    m_min = config["mode_specific_params"]["SHAP"]["SHAP_min_samples"]
    m_max = config["mode_specific_params"]["SHAP"]["SHAP_max_samples"]
    remaining_samples_r1 = len(feature_list) * m_min
    batch_size = config["batch_size"]

    shapley_values = {}
    n_runs, prev_n_runs = 0, 0
    bar = tqdm(total=(m_max)*len(feature_list))

    try:
        # round 1
        while remaining_samples_r1 > 0:
            batch_size = min(batch_size, remaining_samples_r1) # make sure batch size is never bigger than the number of remaining images

            # get the contribution values of batch_size images. this is basically an array of 0/1 where 1=correct prediction, 0 = incorrect prediction
            contribution_dict, n_runs = get_contribution_batch(x, y, model, n_runs, batch_size, feature_list, config)

            # loop over all features to put them in the correct format for the results & calculate (new) variance score from the SHAP paper
            for feature_index, contribution_list in contribution_dict.items():
                if feature_index in shapley_values.keys(): # add values and update var scores
                    shapley_values[feature_index]["values"] = np.hstack((shapley_values[feature_index]["values"], contribution_list))
                    var_score = calculate_var_score(shapley_values[feature_index])
                    shapley_values[feature_index]['var'] = var_score
                    shapley_values[feature_index]['n_samples'] += len(contribution_list)
                else: # create new scores and new var scoress
                    shapley_values[feature_index]= {"values": contribution_list, 'n_samples': len(contribution_list)}
                    var_score = calculate_var_score(shapley_values[feature_index])
                    shapley_values[feature_index]['var'] = var_score
                if config["save_temp_results"] and prev_n_runs + 500 < n_runs: # save about every 500 predictions if save_temp_results is true
                    results = update_results(results, shapley_values)
                    prev_n_runs = n_runs
                bar.n = n_runs
                bar.refresh()
            remaining_samples_r1 -= batch_size

        # round 2
        remaining_samples_r2 = len(feature_list) * (m_max - m_min)
        while remaining_samples_r2 > 0:
            # determine the feature with the highest variance, this is the feature that will be imputed
            feature_index = max(shapley_values, key=lambda x: shapley_values[x]['var'])

            # get the contribution of the coalition
            contribution = get_contribution_single(x, y, feature_list, feature_index, model, config)
            shapley_values[feature_index]['values'] = np.hstack((shapley_values[feature_index]['values'], [contribution]))

            # update the variance score
            var_score = calculate_var_score(shapley_values[feature_index])
            shapley_values[feature_index]['var'] = var_score
            shapley_values[feature_index]['n_samples'] += 1

            # update the remaining samples and continue to the next loop
            remaining_samples_r2 -= 1

            # save the temporary results and update the bar
            if remaining_samples_r2 % 500 == 0:
                bar.n = (len(feature_list) * m_max) - remaining_samples_r2
                bar.refresh()
                if config["save_temp_results"]:
                    results = update_results(results, shapley_values)

        results = save_final_results(results, shapley_values) # save the final results
    except KeyboardInterrupt:
        if config["save_temp_results"]:
            print("KeyboardInterrupt. Saving the results so far")
            results = save_final_results(results, shapley_values) # save the final results
    finally:
        os.environ["warning_displayed"] = "False"


