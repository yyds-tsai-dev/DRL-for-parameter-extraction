import numpy as np
import pandas as pd
import importlib.util
import json

try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(value):
        print(value)


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp950", "big5", "gb18030", "latin1")


def _keras_regularizers():
    try:
        from tensorflow.keras.regularizers import L1, L2, L1L2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for Keras regularizer config.") from exc
    return L1, L2, L1L2


def _keras_constraints():
    try:
        from tensorflow.keras.constraints import MaxNorm, NonNeg, UnitNorm, MinMaxNorm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for Keras constraint config.") from exc
    return MaxNorm, NonNeg, UnitNorm, MinMaxNorm


def _sklearn_metrics():
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for metric config.") from exc
    return mean_absolute_error, mean_squared_error, r2_score


def read_csv_flexible(path, **kwargs):
    """Read CSV files with common UTF and Chinese Windows encodings."""
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs), "utf-8-replace"
    except TypeError:
        if last_error:
            raise last_error
        raise


def read_table_flexible(path, **kwargs):
    """Read CSV or Excel files while preserving the CSV encoding fallback behavior."""
    lower_path = str(path).lower()
    if lower_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kwargs), "excel"
    return read_csv_flexible(path, **kwargs)


def coerce_feature_columns_numeric(df, feature_columns):
    df = df.copy()
    for column in feature_columns:
        if column not in df.columns:
            continue
        if df[column].dtype == bool:
            df[column] = df[column].astype(float)
            continue
        try:
            df[column] = pd.to_numeric(df[column]).astype(float)
        except Exception:
            pass
    return df


def parse_regularizer(reg_config):
    if not reg_config:
        return None
    L1, L2, L1L2 = _keras_regularizers()
    reg_type = reg_config.get("type")
    value = reg_config.get("value")
    if reg_type == "L1":
        return L1(value)
    elif reg_type == "L2":
        return L2(value)
    elif reg_type in {"L1_L2", "L1L2"}:
        return L1L2(l1=reg_config.get("l1_value", 0.0), l2=reg_config.get("l2_value", 0.0))
    else:
        raise ValueError("Regularizer type should be one of 'L1', 'L2', 'L1L2'")

class Config:
    def __init__(self,config):
        self.config = config

        self.INPUT=self.config["Input"]
        self.TRAIN = self.config.get("Training", None)
        self.PREPROCESS = self.config.get("Preprocessing", {})
        self.VALID = self.config.get("Validation", None)
        self.OPTI  = self.config.get("Optimization", None)
        self.SHAP = self.config.get("Shap", None)

        # Input
        self.train_file_input = self.INPUT.get("train_file_input", None)
        self.test_file_input  = self.INPUT.get("test_file_input", None)
        self.features         = self.INPUT.get("feature_file", None)
        self.normalization_method = self.PREPROCESS.get("normalization_method", "z-score")
        
        if self.TRAIN:
            # Training
            self.samples             = self.TRAIN.get("samples", None)
            self.committee_n_jobs    = self.TRAIN.get("committee_n_jobs", None)
            self.tf_intra_op_threads = self.TRAIN.get("tf_intra_op_threads", None)
            self.tf_inter_op_threads = self.TRAIN.get("tf_inter_op_threads", None)
            self.stop_flag_file      = self.TRAIN.get("stop_flag_file", None)
            self.save_models_folder = self.TRAIN.get("save_models_folder", None)
            self.dropout             = self.TRAIN.get("dropout", None)
            self.loss                = self.TRAIN.get("loss", None)

            ## Model
            ### Layers
            model          = self.TRAIN.get("Model", {})
            self.regressors = model.get("Regressors", {})
            if self.regressors.get("type") == "MLP":
                MaxNorm, NonNeg, UnitNorm, MinMaxNorm = _keras_constraints()
                mlp_config = self.regressors.get("MLP", {})
                self.hidden_layer_sizes    = mlp_config.get('hidden_layer_sizes')
                self.activation            = mlp_config.get("activation")
                self.use_bias              = mlp_config.get("use_bias")
                self.kernel_initializer    = mlp_config.get("kernel_initializer")
                self.bias_initializer      = mlp_config.get("bias_initializer")
                self.alpha                 = mlp_config.get("alpha")

                self.kernel_regularizer   = parse_regularizer(mlp_config.get("kernel_regularizer"))
                self.bias_regularizer     = parse_regularizer(mlp_config.get("bias_regularizer"))
                self.activity_regularizer = parse_regularizer(mlp_config.get("activity_regularizer"))
                
                kernel_constraint_config = mlp_config.get("kernel_constraint") or {}
                kernel_constraint_type = kernel_constraint_config.get("type")
                if kernel_constraint_type == "MaxNorm":
                    self.kernel_constraint = MaxNorm(kernel_constraint_config.get("max_value"))
                elif kernel_constraint_type == "NonNeg":
                    self.kernel_constraint = NonNeg()
                elif kernel_constraint_type == "UnitNorm":
                    self.kernel_constraint = UnitNorm()
                elif kernel_constraint_type == "MinMaxNorm":
                    self.kernel_constraint = MinMaxNorm()
                else:
                    self.kernel_constraint = None
                
                bias_constraint_config = mlp_config.get("bias_constraint") or {}
                bias_constraint_type = bias_constraint_config.get("type")
                if bias_constraint_type == "UnitNorm":
                    self.bias_constraint = UnitNorm()
                elif bias_constraint_type == "NonNeg":
                    self.bias_constraint = NonNeg()
                elif bias_constraint_type == "MinMaxNorm":
                    self.bias_constraint = MinMaxNorm()
                elif bias_constraint_type == "MaxNorm":
                    self.bias_constraint = MaxNorm()
                else:
                    self.bias_constraint = None
                
                self.lora_rank = mlp_config.get("lora_rank")

            elif self.regressors.get("type") == "RF":
                rf_config = self.regressors.get("RF", {})
                self.n_estimators              = rf_config.get('n_estimators')
                self.criterion                 = rf_config.get('criterion')
                self.max_depth                 = rf_config.get('max_depth')
                self.min_samples_split        = rf_config.get('min_samples_split')
                self.min_samples_leaf         = rf_config.get('min_samples_leaf')
                self.min_weight_fraction_leaf = rf_config.get('min_weight_fraction_leaf')
                self.max_features             = rf_config.get('max_features')
                self.max_leaf_nodes           = rf_config.get('max_leaf_nodes')
                self.min_impurity_decrease    = rf_config.get('min_impurity_decrease')
                self.bootstrap                = rf_config.get('bootstrap')
                self.oob_score                = rf_config.get('oob_score')
                self.n_jobs                   = rf_config.get('n_jobs')
                self.random_state             = rf_config.get('random_state')
                self.rf_verbose               = rf_config.get('verbose')
                self.warm_start               = rf_config.get('warm_start')
                self.ccp_alpha                = rf_config.get('ccp_alpha')
                self.max_samples              = rf_config.get('max_samples')
                self.monotonic_cst            = rf_config.get('monotonic_cst')

            elif self.regressors.get("type") == "XGB":
                xgb_config = self.regressors.get("XGB", {})
                self.n_estimators          = xgb_config.get('n_estimators')
                self.max_depth             = xgb_config.get('max_depth')
                self.grow_policy           = xgb_config.get('grow_policy')
                self.XGB_learning_rate     = xgb_config.get('XGB_learning_rate')
                self.verbosity             = xgb_config.get('verbosity')
                self.objective             = xgb_config.get('objective')
                self.booster               = xgb_config.get('booster')
                self.tree_method           = xgb_config.get('tree_method')
                self.n_jobs                = xgb_config.get('n_jobs')
                self.gamma                 = xgb_config.get('gamma')
                self.min_child_weight      = xgb_config.get('min_child_weight')
                self.max_delta_step        = xgb_config.get('max_delta_step')
                self.subsample             = xgb_config.get('subsample')
                self.sampling_method       = xgb_config.get('sampling_method')
                self.colsample_bytree      = xgb_config.get('colsample_bytree')
                self.colsample_bylevel     = xgb_config.get('colsample_bylevel')
                self.colsample_bynode      = xgb_config.get('colsample_bynode')
                self.reg_alpha             = xgb_config.get('reg_alpha')
                self.reg_lambda            = xgb_config.get('reg_lambda')
                self.scale_pos_weight      = xgb_config.get('scale_pos_weight')
                self.base_score            = xgb_config.get('base_score')
                self.random_state          = xgb_config.get('random_state')
                self.missing               = xgb_config.get('missing')
                self.num_parallel_tree     = xgb_config.get('num_parallel_tree')
                self.monotone_constraints  = xgb_config.get('monotone_constraints')
                self.interaction_constraints = xgb_config.get('interaction_constraints')
                self.importance_type       = xgb_config.get('importance_type')
                self.device                = xgb_config.get('device')
                self.validate_parameters   = xgb_config.get('validate_parameters')
                self.enable_categorical    = xgb_config.get('enable_categorical')
                self.feature_types         = xgb_config.get('feature_types')
                self.max_cat_to_onehot     = xgb_config.get('max_cat_to_onehot')
                self.max_cat_threshold     = xgb_config.get('max_cat_threshold')
                self.multi_strategy        = xgb_config.get('multi_strategy')

                eval_metric = xgb_config.get('eval_metric')
                if eval_metric == "mean_squared_error":
                    _, mean_squared_error, _ = _sklearn_metrics()
                    self.eval_metric = mean_squared_error
                elif eval_metric == "mean_absolute_error":
                    mean_absolute_error, _, _ = _sklearn_metrics()
                    self.eval_metric = mean_absolute_error
                elif eval_metric == "r2_score":
                    _, _, r2_score = _sklearn_metrics()
                    self.eval_metric = r2_score
                else:
                    self.eval_metric = None

                self.early_stopping_rounds = xgb_config.get('early_stopping_rounds')
                self.XGB_callbacks         = xgb_config.get('XGB_callbacks')

            elif self.regressors.get("type") == "SVR":
                svr_config = self.regressors.get("SVR", {})
                self.svr_kernel = svr_config.get("kernel", "rbf")
                self.svr_C = svr_config.get("C", 1.0)
                self.svr_epsilon = svr_config.get("epsilon", 0.1)
                self.svr_gamma = svr_config.get("gamma", "scale")

            elif self.regressors.get("type") == "KNN":
                knn_config = self.regressors.get("KNN", {})
                self.knn_n_neighbors = knn_config.get("n_neighbors", 5)
                self.knn_weights = knn_config.get("weights", "uniform")
                self.knn_algorithm = knn_config.get("algorithm", "auto")
                self.knn_p = knn_config.get("p", 2)
                self.n_jobs = knn_config.get("n_jobs", None)

            elif self.regressors.get("type") == "GBR":
                gbr_config = self.regressors.get("GBR", {})
                self.gbr_n_estimators = gbr_config.get("n_estimators", 100)
                self.gbr_learning_rate = gbr_config.get("learning_rate", 0.1)
                self.gbr_max_depth = gbr_config.get("max_depth", 3)
                self.gbr_subsample = gbr_config.get("subsample", 1.0)
                self.random_state = gbr_config.get("random_state", 42)

            elif self.regressors.get("type") == "GPR":
                gpr_config = self.regressors.get("GPR", {})
                self.gpr_alpha = gpr_config.get("alpha", 1e-10)
                self.gpr_length_scale = gpr_config.get("length_scale", 1.0)
                self.gpr_constant_value = gpr_config.get("constant_value", 1.0)
                self.gpr_normalize_y = gpr_config.get("normalize_y", True)
                self.gpr_n_restarts_optimizer = gpr_config.get("n_restarts_optimizer", 0)
                self.random_state = gpr_config.get("random_state", 42)

            elif self.regressors.get("type") == "KMeans":
                kmeans_config = self.regressors.get("KMeans", {})
                self.n_clusters = kmeans_config.get("n_clusters", 3)
                self.kmeans_init = kmeans_config.get("init", "k-means++")
                self.kmeans_n_init = kmeans_config.get("n_init", 10)
                self.kmeans_max_iter = kmeans_config.get("max_iter", 300)
                self.random_state = kmeans_config.get("random_state", 42)

            elif self.regressors.get("type") == "MLP_CLS":
                mlp_cls_config = self.regressors.get("MLP_CLS", {})
                self.mlp_cls_hidden_layer_sizes = mlp_cls_config.get("hidden_layer_sizes", [100])
                self.mlp_cls_activation = mlp_cls_config.get("activation", "relu")
                self.mlp_cls_solver = mlp_cls_config.get("solver", "adam")
                self.mlp_cls_learning_rate = mlp_cls_config.get("learning_rate", 0.001)
                self.mlp_cls_batch_size = mlp_cls_config.get("batch_size", "auto")
                self.mlp_cls_max_iter = mlp_cls_config.get("max_iter", 200)
                self.random_state = mlp_cls_config.get("random_state", 42)

            elif self.regressors.get("type") == "Hierarchical":
                hierarchical_config = self.regressors.get("Hierarchical", {})
                self.n_clusters = hierarchical_config.get("n_clusters", 3)
                self.hierarchical_linkage = hierarchical_config.get("linkage", "ward")
                self.hierarchical_metric = hierarchical_config.get("metric", "euclidean")

            elif self.regressors.get("type") == "GPC":
                gpc_config = self.regressors.get("GPC", {})
                self.gpc_length_scale = gpc_config.get("length_scale", 1.0)
                self.gpc_constant_value = gpc_config.get("constant_value", 1.0)
                self.gpc_max_iter_predict = gpc_config.get("max_iter_predict", 100)
                self.gpc_n_restarts_optimizer = gpc_config.get("n_restarts_optimizer", 0)
                self.random_state = gpc_config.get("random_state", 42)

            ### Optimizer
            optimizer      = model.get("Optimizer", {})
            self.opt_type  = optimizer.get("opt_type")
            self.SGD       = optimizer.get("SGD", {})
            self.Adam      = optimizer.get("Adam", {})
            self.AdamW     = optimizer.get("AdamW", {})
            self.RMSprop   = optimizer.get("RMSprop", {})

            if self.opt_type == "SGD":
                SGD = self.SGD
                self.learning_rate = SGD.get("learning_rate")
                self.momentum = SGD.get("momentum")
                self.nesterov = SGD.get("nesterov")
                self.weight_decay = SGD.get("weight_decay")
                self.clipnorm = SGD.get("clipnorm")
                self.clipvalue = SGD.get("clipvalue")
                self.global_clipnorm = SGD.get("global_clipnorm")
                self.use_ema = SGD.get("use_ema")
                self.ema_momentum = SGD.get("ema_momentum")
                self.ema_overwrite_frequency = SGD.get("ema_overwrite_frequency")
                self.loss_scale_factor = SGD.get("loss_scale_factor")
                self.gradient_accumulation_steps = SGD.get("gradient_accumulation_steps")

            elif self.opt_type == "Adam":
                Adam = self.Adam
                self.learning_rate = Adam.get("learning_rate")
                self.beta_1 = Adam.get("beta_1")
                self.beta_2 = Adam.get("beta_2")
                self.epsilon = Adam.get("epsilon")
                self.amsgrad = Adam.get("amsgrad")
                self.weight_decay = Adam.get("weight_decay")
                self.clipnorm = Adam.get("clipnorm")
                self.clipvalue = Adam.get("clipvalue")
                self.global_clipnorm = Adam.get("global_clipnorm")
                self.use_ema = Adam.get("use_ema")
                self.ema_momentum = Adam.get("ema_momentum")
                self.ema_overwrite_frequency = Adam.get("ema_overwrite_frequency")
                self.loss_scale_factor = Adam.get("loss_scale_factor")
                self.gradient_accumulation_steps = Adam.get("gradient_accumulation_steps")

            elif self.opt_type == "AdamW":
                adamw_opts = self.AdamW
                self.learning_rate = adamw_opts.get("learning_rate")
                self.beta_1 = adamw_opts.get("beta_1")
                self.beta_2 = adamw_opts.get("beta_2")
                self.epsilon = adamw_opts.get("epsilon")
                self.amsgrad = adamw_opts.get("amsgrad")
                self.weight_decay = adamw_opts.get("weight_decay")
                self.clipnorm = adamw_opts.get("clipnorm")
                self.clipvalue = adamw_opts.get("clipvalue")
                self.global_clipnorm = adamw_opts.get("global_clipnorm")
                self.use_ema = adamw_opts.get("use_ema")
                self.ema_momentum = adamw_opts.get("ema_momentum")
                self.ema_overwrite_frequency = adamw_opts.get("ema_overwrite_frequency")
                self.loss_scale_factor = adamw_opts.get("loss_scale_factor")
                self.gradient_accumulation_steps = adamw_opts.get("gradient_accumulation_steps")

            elif self.opt_type == "RMSprop":
                rmsprop_opts = self.RMSprop
                self.learning_rate = rmsprop_opts.get("learning_rate")
                self.rho = rmsprop_opts.get("rho")
                self.momentum = rmsprop_opts.get("momentum")
                self.epsilon = rmsprop_opts.get("epsilon")
                self.centered = rmsprop_opts.get("centered")
                self.weight_decay = rmsprop_opts.get("weight_decay")
                self.clipnorm = rmsprop_opts.get("clipnorm")
                self.clipvalue = rmsprop_opts.get("clipvalue")
                self.global_clipnorm = rmsprop_opts.get("global_clipnorm")
                self.use_ema = rmsprop_opts.get("use_ema")
                self.ema_momentum = rmsprop_opts.get("ema_momentum")
                self.ema_overwrite_frequency = rmsprop_opts.get("ema_overwrite_frequency")
                self.loss_scale_factor = rmsprop_opts.get("loss_scale_factor")
                self.gradient_accumulation_steps = rmsprop_opts.get("gradient_accumulation_steps")

            ### Fit
            Fit = model.get("Fit", {})
            self.batch_size = Fit.get("batch_size")
            self.epochs = Fit.get("epochs")
            self.verbose = Fit.get("verbose")
            self.callbacks = Fit.get("callbacks")
            self.validation_split = Fit.get("validation_split")
            self.validation_data = Fit.get("validation_data")
            self.shuffle = Fit.get("shuffle")
            self.class_weight = Fit.get("class_weight")
            self.sample_weight = Fit.get("sample_weight")
            self.initial_epoch = Fit.get("initial_epoch")
            self.steps_per_epoch = Fit.get("steps_per_epoch")
            self.validation_steps = Fit.get("validation_steps")
            self.validation_batch_size = Fit.get("validation_batch_size")
            self.validation_freq = Fit.get("validation_freq")
        if self.VALID:
            # Validation
            self.load_models_folder = self.VALID.get("load_models_folder")
            self.save_results_folder = self.VALID.get("save_results_folder")
            self.fig_file_folder = self.VALID.get("fig_file_folder")
        if self.OPTI:
            # Optimization
            self.TARGET_PROPERTY = self.OPTI.get("TARGET_PROPERTY") 
            self.STEP_SZ_rate = self.OPTI.get("STEP_SZ_rate")
            self.LWBD_rate = self.OPTI.get("LWBD_rate") 
            self.UPBD_rate = self.OPTI.get("UPBD_rate")
            self.perturbation_max_steps = self.OPTI.get("perturbation_max_steps") 
            self.boltz_alpha = self.OPTI.get("boltz_alpha") 
            self.boltz_min_t = self.OPTI.get("boltz_min_t") 
            self.max_steps = self.OPTI.get("max_steps") 
            self.extend_steps = self.OPTI.get("extend_steps") 
            self.save_opt_folder = self.OPTI.get("save_opt_folder")
        if self.SHAP:
            # Shap
            self.load_models_folder = self.SHAP.get("load_models_folder")
            self.save_results_folder = self.SHAP.get("save_results_folder")
            self.force_plot_featrues = self.SHAP.get("force_plot_features")
            self.interaction_index = self.SHAP.get("dependence_plot_interaction_index")

        self.df_idx = None
        self.data_x = None
        self.data_y = None

    def read_features(self):
        if str(self.features).lower().endswith(".json"):
            with open(self.features, "r", encoding="utf-8") as f:
                self.features = json.load(f)
        else:
            module_name = self.features.replace("/", ".").replace(".py", "")
            module = importlib.import_module(module_name)
            self.features = module.features
        print("=======================================")
        print("self.features:",self.features)
        print("=======================================")
        

    def read_train_data(self):

        self.df, self.train_file_encoding = read_csv_flexible(self.train_file_input)
        print("=======================================")
        print("## selected features are ##")
        print("=======================================")
        # pdb.set_trace()
        self.x_col = self.features["features"]
        self.y_col = self.features["target"]
        self.df = self.df.dropna(subset=self.x_col + self.y_col)
        self.df = coerce_feature_columns_numeric(self.df, self.x_col)
        if self.df.empty:
            raise ValueError("No rows remain after removing missing values in selected feature/target columns.")
        self.x = self.df[self.x_col]
        display(self.x.head())

        print("=======================================")
        print("## selected target is ##")
        print("=======================================")
        self.y = self.df[self.y_col]
        display(self.y.head())
        
        print("X-dim: {}, Y-dim: {}".format(self.x.shape, self.y.shape))
        # import pdb
        # pdb.set_trace()
        # Get features & targets.
        self.FEATS =self.x_col
        self.TARGETS = self.y_col

        # Packing the features & targets into dictionary.
        self.data = [{'x':x, 'y':y} for x, y in zip(self.df[self.FEATS].to_numpy(), self.df[self.TARGETS].to_numpy())]

        # Generalized getter for training a committee.
        self.get_x = lambda d: d['x']
        self.get_y = lambda d: d['y']
        
        # Prepare indicies for bootstrap sampling.
        self.df_idx = pd.DataFrame([i for i in range(len(self.data))])
        
        # Extract all feature/target paris.
        self.data_x = np.asarray([self.get_x(mt) for mt in self.data])
        self.data_y = np.asarray([self.get_y(mt) for mt in self.data])

        # return self.data_x, self.data_y, self.df_idx
    def read_test_data(self, mode="test"):

        self.df, self.test_file_encoding = read_csv_flexible(self.test_file_input)
        print("=======================================")
        print("## selected features are ##")
        print("=======================================")

        self.x_col = self.features["features"]
        self.y_col = self.features["target"]
        if mode == "test":
            self.df = self.df.dropna(subset=self.x_col + self.y_col)
        else:
            self.df = self.df.dropna(subset=self.x_col)
        self.df = coerce_feature_columns_numeric(self.df, self.x_col)
        if self.df.empty:
            raise ValueError("No rows remain after removing missing values in selected columns.")
        self.x = self.df[self.x_col]
        display(self.x.head())

        
        print("=======================================")
        print("## selected target is ##")
        print("=======================================")
        if mode == "test":
            self.y = self.df[self.y_col]
            display(self.y.head())
            print("X-dim: {}, Y-dim: {}".format(self.x.shape, self.y.shape))
        else:
            self.y = None
            print("X-dim: {}".format(self.x.shape))
        
        # Get features & targets.
        self.FEATS =self.x_col
        self.TARGETS = self.y_col

        # Packing the features & targets into dictionary.
        if self.y is not None:
            self.data = [{'x': x, 'y': y} for x, y in zip(self.df[self.FEATS].to_numpy(), self.df[self.TARGETS].to_numpy())]
        else:
            self.data = [{'x': x, 'y': None} for x in self.df[self.FEATS].to_numpy()]
        
        # Generalized getter for training a committee.
        self.get_x = lambda d: d['x']
        self.get_y = lambda d: d['y']
        
        # Prepare indicies for bootstrap sampling.
        self.df_idx = pd.DataFrame([i for i in range(len(self.data))])
        
        # Extract all feature/target paris.
        self.data_x = np.asarray([self.get_x(mt) for mt in self.data])
        self.data_y = np.asarray([self.get_y(mt) for mt in self.data]) if self.y is not None else None

        # return self.data_x, self.data_y, self.df_idx

def load_json(file_name):
    """ Return python object stored in pickle format.
    Args:
        file_name: saved pickle.
    """
    import json
    with open(file_name, 'r') as f:
        data = json.load(f)
    
    return data
