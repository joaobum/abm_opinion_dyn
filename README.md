# A co-evolution model for opinions in a social network
## An agent-based proposal for collective beliefs shaping and being shaped by the social structure


### Model running instructions
1. Install all dependencies by running:

`pip install -r requirements.txt`

2. Set the running parameters and results folder in configuration.py

3. Set the running mode in main, one of: MODEL_TEST, PARAMETER_SWEEP or STATISTICAL_ANALYSIS

4. Execute the main file:

`python main.py`

### Result analysis instructions
1. Execute the analysis module:

`python analysis.py`

2. Follow the CLI instructions. Filter are based on filename, and can be set in the following example format:

`set ag=200*po=3`

3. Once the list is displayed, plots can either be shown in a browser or sved to html file. Both plotting and saving work with a range of indexes. To plot indexes 5 to 10:

`plot 5-10`
