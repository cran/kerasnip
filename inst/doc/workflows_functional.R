## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = reticulate::py_module_available("keras")
)
# Suppress verbose Keras output for the vignette
options(keras.fit_verbose = 0)
set.seed(123)

## ----load-packages------------------------------------------------------------
library(kerasnip)
library(tidymodels)
library(keras3)
library(dplyr) # For data manipulation
library(ggplot2) # For plotting
library(future) # For parallel processing
library(finetune) # For racing

## ----data-prep----------------------------------------------------------------
# Select relevant columns and remove rows with missing values
ames_df <- ames |>
  select(
    Sale_Price,
    Gr_Liv_Area,
    Year_Built,
    Neighborhood,
    Bldg_Type,
    Overall_Cond,
    Total_Bsmt_SF,
    contains("SF")
  ) |>
  na.omit()

# Split data into training and testing sets
set.seed(123)
ames_split <- initial_split(ames_df, prop = 0.8, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# Create cross-validation folds for tuning
ames_folds <- vfold_cv(ames_train, v = 5, strata = Sale_Price)

## ----create-recipe------------------------------------------------------------
ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) |>
  step_normalize(all_numeric_predictors()) |>
  step_collapse(all_numeric_predictors(), new_col = "numerical_input") |>
  step_dummy(Neighborhood) |>
  step_collapse(starts_with("Neighborhood"), new_col = "neighborhood_input") |>
  step_dummy(Bldg_Type) |>
  step_collapse(starts_with("Bldg_Type"), new_col = "bldg_input") |>
  step_dummy(Overall_Cond) |>
  step_collapse(starts_with("Overall_Cond"), new_col = "condition_input")

## ----define-kerasnip-model----------------------------------------------------
# Define layer blocks for multi-input functional model

# Input blocks for numerical and categorical features
input_numerical <- function(input_shape) {
  layer_input(shape = input_shape, name = "numerical_input")
}

input_neighborhood <- function(input_shape) {
  layer_input(shape = input_shape, name = "neighborhood_input")
}

input_bldg <- function(input_shape) {
  layer_input(shape = input_shape, name = "bldg_input")
}

input_condition <- function(input_shape) {
  layer_input(shape = input_shape, name = "condition_input")
}

# Processing blocks for each input type
dense_numerical <- function(tensor, units = 32, activation = "relu") {
  tensor |>
    layer_dense(units = units, activation = activation)
}

dense_categorical <- function(tensor, units = 16, activation = "relu") {
  tensor |>
    layer_dense(units = units, activation = activation)
}

# Concatenation block
concatenate_features <- function(numeric, neighborhood, bldg, condition) {
  layer_concatenate(list(numeric, neighborhood, bldg, condition))
}

# Output block for regression
output_regression <- function(tensor) {
  layer_dense(tensor, units = 1, name = "output")
}

# Create the kerasnip model specification function
create_keras_functional_spec(
  model_name = "ames_functional_mlp",
  layer_blocks = list(
    numerical_input = input_numerical,
    neighborhood_input = input_neighborhood,
    bldg_input = input_bldg,
    condition_input = input_condition,
    processed_numerical = inp_spec(dense_numerical, "numerical_input"),
    processed_neighborhood = inp_spec(dense_categorical, "neighborhood_input"),
    processed_bldg = inp_spec(dense_categorical, "bldg_input"),
    processed_condition = inp_spec(dense_categorical, "condition_input"),
    combined_features = inp_spec(
      concatenate_features,
      c(
        numeric = "processed_numerical",
        neighborhood = "processed_neighborhood",
        bldg = "processed_bldg",
        condition = "processed_condition"
      )
    ),
    output = inp_spec(output_regression, "combined_features")
  ),
  mode = "regression"
)

## ----define-tune-spec---------------------------------------------------------
# Define the tunable model specification
functional_mlp_spec <- ames_functional_mlp(
  # Tunable parameters for numerical branch
  processed_numerical_units = tune(),
  # Tunable parameters for categorical branch
  processed_neighborhood_units = tune(),
  processed_bldg_units = tune(),
  processed_condition_units = tune(),
  # Fixed compilation and fitting parameters
  compile_loss = "mean_squared_error",
  compile_optimizer = "adam",
  compile_metrics = c("mean_absolute_error"),
  fit_epochs = 50,
  fit_batch_size = 32,
  fit_validation_split = 0.2,
  fit_callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5)
  )
) |>
  set_engine("keras")

print(functional_mlp_spec)

## ----create-workflow----------------------------------------------------------
ames_wf <- workflow() |>
  add_recipe(ames_recipe) |>
  add_model(functional_mlp_spec)

print(ames_wf)

## ----create-tuning-grid-------------------------------------------------------
# Define the tuning grid
params <- extract_parameter_set_dials(ames_wf) |>
  update(
    processed_numerical_units = hidden_units(range = c(32, 128)),
    processed_neighborhood_units = hidden_units(range = c(16, 64)),
    processed_bldg_units = hidden_units(range = c(16, 64)),
    processed_condition_units = hidden_units(range = c(16, 64))
  )
functional_mlp_grid <- grid_regular(params, levels = 3)

print(functional_mlp_grid)

## ----tune-model, cache=TRUE---------------------------------------------------
# Note: Parallel processing with `plan(multisession)` is currently not working
# with Keras models due to backend conflicts

set.seed(123)
ames_tune_results <- tune_race_anova(
  ames_wf,
  resamples = ames_folds,
  grid = functional_mlp_grid,
  metrics = metric_set(rmse, mae, rsq),
  control = control_race(save_pred = TRUE, save_workflow = TRUE)
)

## ----inspect-results----------------------------------------------------------
# Show the best performing models based on RMSE
show_best(ames_tune_results, metric = "rmse", n = 5)

# Autoplot the results
# Currently does not work due to a label issue: autoplot(ames_tune_results)

# Select the best hyperparameters
best_functional_mlp_params <- select_best(ames_tune_results, metric = "rmse")
print(best_functional_mlp_params)

## ----finalize-fit-------------------------------------------------------------
# Finalize the workflow with the best hyperparameters
final_ames_wf <- finalize_workflow(ames_wf, best_functional_mlp_params)

# Fit the final model on the full training data
final_ames_fit <- fit(final_ames_wf, data = ames_train)

print(final_ames_fit)

## ----inspect-final-keras-model-summary----------------------------------------
# Extract the Keras model summary
final_ames_fit |>
  extract_fit_parsnip() |>
  extract_keras_model() |>
  summary()

## ----inspect-final-keras-model-plot, eval=FALSE-------------------------------
# # Plot the Keras model
# final_ames_fit |>
#   extract_fit_parsnip() |>
#   extract_keras_model() |>
#   plot(show_shapes = TRUE)

## ----inspect-final-keras-model-history----------------------------------------
# Plot the training history
final_ames_fit |>
  extract_fit_parsnip() |>
  extract_keras_history() |>
  plot()

## ----predict-evaluate---------------------------------------------------------
# Make predictions on the test set
ames_test_pred <- predict(final_ames_fit, new_data = ames_test)

# Combine predictions with actuals
ames_results <- tibble::tibble(
  Sale_Price = ames_test$Sale_Price,
  .pred = ames_test_pred$.pred
)

print(head(ames_results))

# Evaluate performance using yardstick metrics
metrics_results <- metric_set(
  rmse,
  mae,
  rsq
)(
  ames_results,
  truth = Sale_Price,
  estimate = .pred
)

print(metrics_results)

