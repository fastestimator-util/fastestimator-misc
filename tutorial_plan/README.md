discuss part:
1. hierarchy of beginner, advanced, FAQ
2. scheduler in beginner or advanced
3.
: diretory
*: jupyter notebook file
-: heading in jupyter notebook
--------------------------------------------------------------------------------------
tutorial structure
* overview
(content)
[* related apphub]
---------------------------------------------------------------------------------------
beginner
    1. Getting started (MNIST image classification)
        - Pipeline -> Network -> Estimator
        - Ops
            - Concept (How things connect? (input, output mechanism)
            - mode meaning
            - mode expression
        - trace
            - Concept (inputs, outputs)
            - mode   (act like callbacks, reference later for more capabilities)
         Example: ModelSaver+Accuracy   


    2. FastEstimator dataset
        - how to create dataset (simple)
            - Data from disk
                - LabeledDirDataset
                - CSV dataset
            - Data from memory
                - NumpyDataset
            - import FE dataset (public dataset)
        - How to split dataset (simple)
        - len(dataset), dataset[0]


    3. Create data pipeline
        - Pipeline data source
            - tf.data.Dataset (example)
            - pytorch dataloader (example)
            - FE dataset, torch dataset
        - Pipeline steps
            - NumpyOp 
                - univariate
                - multivariate(why we structure them this way- ref albumentation)
                - meta
            - mention customization
            - Customize NumpyOp(simple example)
        - Pipeline other args
            - batch_size
            - train_data, eval_data, test_data
        - Piepline debugging
            - get_result


    4. Create model
        - Define model function(pytorch, tensorflow)
            - import FE architecture
            - import tf.keras.application
            - import torchvision pretrained model
        - Compile model
            - fe.build
            - Define optimizer
                - string
                - lambda function
            - load model weight
            - model name

    5. Create network
        - Netowrk component
            - TensorOp
                - ModelOp
                - UpdateOp
                - Loss
            - CustomizeTensorOp(simple example)


    6. Create estimator
        - Monitor training process
            - Trace:
                - metric
                - logger
                - model saver
        - Estimator other args
            - max_steps_per_epoch
            - log_steps
            - monitor_names
        - Trigger fit, test


    7. Inference
        -data[0]
        - pipeline transform
        - network transform



advanced
    * CLI usage:
        -(cli: looking for get_estimator,  --arg is replacing the arg in get_estimator)
        - fastestiamtor train
        - fastestiamtor test
        - fastestiamtor train --hyperparameters param.json

   * Op
        - Deep go through of Operator funcition
            - state
            - data
        - NumpyOp:
            - deleteOp
            - Customize numpyOp

        -TensorOp:
            - why we don't need `DeleteOp` - we already have gpu key filtering in network
           - customize tensorOp
           - backend


    FE dataset
        * how to customize dataset (point to pytorch tutorial
        * how to split
            - based on sample index
            - probability
            - number
            - muti-split (multiple of above)
        * summary
            - ds.summary
            - print(ds)
        * global dataset editing:
            - apply sk.standardize (apphub tabular) ?
        * how to create dataset from disk
            - LabeledDirDataset
            - CSV dataset
            - generator_dataset
            - DirDataset - (unlabeledDataset)
            - LabeledDirDataset
            - generator_dataset
        * batch dataset
            - deterministic batching: batch=8, 5 postive, 3 negative (same keys)
            - stochastic batching: batch=8, 0.8 from postive, 0.2 from negative
            - unpaired dataset batch=2, one from horse, one from zebra (different keys)


    * Pipeline:
        - pipeline padding
        - get_loader: (loop through dataset)
        - benchmark
    



    Scheduler:
        * Scheduler Basics: (epoch is unit time for scheduler)
            - EpochScheduler: {epoch: content} : {1: x, 3: None 4: y}:
                -number means epoch of change.
                - use None as op if no op
            - RepeatScheduler: [list of content] one example: intermitent adversarial training
        * things you can schedule:
            - Pipeline:
                - schedule data: train_data, eval_data: (transfer learning)
                - batch_size
                - ops
            - Network:
                - ops
                - optimizer


    Summary
       * Accessing history in python way
            summary = est.fit(summary="experiment1")
            summary = est.test(summary="experiment1")

       * TensorBoard
        
       * log parsing &experiment logging (visualization):
            - use summary object
            - use trainig log txt

    Trace
        * How multiple metrics work together
            - ( T1: accuracy, T2: f1_score, )
        * How to customize Trace:
            - for debugging, monitoring/ calculating metrics
            - trace time point (epoch_start, batch_start ....)


    * learning rate scheduling
        - provide lambda function, then use 'epoch' or 'step' as argument name
        - use exsiting lr shceudler(cosine/cyclic coscine/ linear decay)


     XAI
        * Saliency tutorial


FAQ
    * how to save training model
        - Save by frequency
        - Save by metircs(best)
    * how to load model weight