# Ripple

Ripple is a serverless computing framework designed to make complex applications on AWS Lambda.
Ripple allows the user to chain lambda functions using S3 triggers 

## Setup
Ripple works by chaining a series of Lambda functions together.
We'll walk through one simple Ripple example.

### Copy File
The first example will simply copy the input file that triggered the pipeline.
To start, run `simple.py` in the examples folder.
This file sets up a pipeline that calls an application called `echo`.
`echo.py` can be found in the `applications` folder.
It takes an input file that we specify to be new line delimited and writes it out to a new file.
When ``simple.py`` is run, the pipeline is created and uploadedto Lambda.
The pipeline is also written to a JSON file called `simple.json` to the json folder.
The JSON file will look like the following.

```
{
  "bucket": "maccoss-tide",
  "log": "maccoss-log",
  "timeout": 600,
  "functions": {
    "echo": {
      "application": "echo",
      "file": "application",
      "input_format": "new_line",
      "memory_size": 1024,
      "output_format": null,
      "formats": [
        "new_line"
      ],
      "imports": []
    }
  },
  "pipeline": [
    {
      "name": "echo"
    }
  ],
  "region": "us-west-2",
  "role": "service-role/lambdaFullAccessRole"
}
```

The `functions` part is the set of all Lambda functions needed by the pipeline.
A function can be used multiple times in the pipeline.
This section specifies parameters that are applicable to every instance of the function, such as the amount of allocated memory or the file format.

The `pipeline` part specifies the order of execution of the Lambda functions.
The parameters in the section are for only instances that occur in this stage of the pipeline.
For example, you may want to call the same combine function twice in the pipeline, but only have one of them sort the output.

The files written will have the format `<prefix>/<timestamp>-<nonce>/<bin_id>-<num_bins>/<file_id>-<execute>-<num_files>.<ext>`.

The prefix indicates the stage of the of the pipeline that wrote the file.
In the above example, the input that triggers the pipeline will have the prefix "0" and the output from the echo function will have the prefix "1".
The timestamp indicates when a given run was instantiated and the nonce is an identifier for the run.
The bin ID specifies which bin the file is in. The bins are used to sort and combine subsets of data.
Each file in a bin is given a number between 1 and the number of files in the bin.
The execute value is used if we need to force a function to re-execute or to tell a function not to execute (this is used for deadline scheduling).

## Trigger Pipeline
To upload a file to a Lambda pipeline, run:
```
python3 upload.py --destination_bucket_name <bucket-used-for-application> --key <name-of-file-to-upload> [--source_bucket_name <s3-bucket-input-file-is-located-in>
```
A user can either upload a file from their computer or from S3, however using the upload script is easier, as it handles formatting the file name.

## Functions
```
input.combine(params={}, config={})
```
Combines the output from the previous stage into one file.<br/>
**Parameters**
1. **params**: Parameters to pass in to the application call.
2. **config**: Configuration for the function such as memory size.

```
input.map(table, func, params={}, config={})
```
Applies the function specified by `func` to each item in `table`.<br/>
**Parameters**
1. **table**: Name of table we're mapping over.
2. **func**: Lambda function to apply to each item in `table`.
  The lambda function should take the name of the input key and item key as parameters.<br/>
  ``lambda input_key, bucket_key: input_key.run("train", params={"train_data": bucket_key, "test_data": input_key})``
3. **params**: Parameters to pass in to the application call.
4. **config**: Configuration for the function such as memory size.

```
input.run(application_name, params={}, config={})
```
Runs the application file from the applications folder named `application-name`.<br/>
**Parameters**
1. **application_name**: Name of the application file to run.
2. **params**: Parameters to pass in to the application call.
3. **config**: Configuration for the function such as memory size.

```
input.sort(identifier, params={}, config={})
```
Sorts the input based on the identifier.<br/>
**Parameters**
1. **identifier**: Identifier to sort input by.
2. **params**: Parameters to pass in to the application call.
3. **config**: Configuration for the function such as memory size.

```
input.split(params={}, config={})
```
Splits the input into multiple chunks to be analyzed in parallel. By default, splits into about 100MB segments.
The user can change this by specifying `split_size` in `params`.<br/>
**Parameters**
1. **params**: Parameters to pass in to the application call.
2. **config**: Configuration for the function such as memory size.

```
input.top(identifier, number, params={}, config={})
```
Returns the top `number` of items based on the value specified by the `identifier`.<br/>
**Parameters**
1. **identifier**: Identifier to sort input by.
2. **number**: Number of top items to return.
3. **params**: Parameters to pass in to the application call.
4. **config**: Configuration for the function such as memory size.
