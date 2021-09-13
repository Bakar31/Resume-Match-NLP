[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBakar31%2FHire-The-Perfect-Candidate&count_bg=%2379C83D&title_bg=%231124B6&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Hire-The-Perfect-Candidate(The Perfect Fit)
## Platform: HackerEarth
## Rank: 9th among 2381 participates.

## Data description:

<p>The <em>dataset</em> folder contains the following files:</p>

<ul>
	<li><em>train.csv</em>: 90 x 2</li>
	<li><em>test.csv</em>: 60 x 1</li>
  <li><em>trainResumes</em>:  90 resumes that you must use for training model</li>
	<li><em>testResumes</em>: 60 resumes that you must use for testing model</li>
  <li><em>Job description.pdf</em>: PDF file that represents the job description of a Machine Learning engineer</li>
  
</ul>

## Evaluation metric:
score = 100*max(0, 1 - metrics.mean_squared_log_error(actual, predicted))

## Best Model:
LightBGM algorithm with counvectorized data has the highest score.
