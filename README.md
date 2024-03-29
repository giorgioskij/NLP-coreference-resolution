# NLP 2022: Homework #3

This is the third homework of the NLP 2022 course at Sapienza University of Rome.

# Results:
The file report.pdf contains a report of my project and its results.

#### Instructor

* **Roberto Navigli**
  * Webpage: [http://www.diag.uniroma1.it/~navigli/](http://www.diag.uniroma1.it/~navigli)

#### Teaching Assistants

* **Andrei Stefan Bejgu**
* **Riccardo Orlando**
* **Alessandro Scirè**
* **Simone Tedeschi**

#### Course Info

* [http://naviglinlp.blogspot.com/](http://naviglinlp.blogspot.com/)

## Requirements

* Ubuntu distribution
  * Either 22.04 or the current LTS (20.04) are perfectly fine
  * If you do not have it installed, please use a virtual machine (or install it as your secondary OS). Plenty of tutorials online for this part
* [conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community

## Notes

Unless otherwise stated, all commands here are expected to be run from the root directory of this project

## Setup Environment

As mentioned in the slides, differently from previous years, this year we will be using Docker to remove any issue pertaining your code runnability. If test.sh runs
on your machine (and you do not edit any uneditable file), it will run on ours as well; we cannot stress enough this point.

Please note that, if it turns out it does not run on our side, and yet you claim it run on yours, the **only explanation** would be that you edited restricted files,
messing up with the environment reproducibility: regardless of whether or not your code actually runs on your machine, if it does not run on ours,
you will be failed automatically. **Only edit the allowed files**.

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

For those interested, *test.sh* essentially setups a server exposing your model through a REST Api and then queries this server, evaluating your model.

### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding.
For those who might be unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependecies to run. We will be using conda to create the environment for this client.

```bash
conda create -n nlp2022-hw3 python=3.9
conda activate nlp2022-hw3
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp2022-hw3
bash test.sh data/dev.tsv
```

Actually, you can replace *data/dev.tsv* to point to a different file, as far as the target file has the same format.

If you hadn't changed *hw3/stud/model.py* yet when you run test.sh, the scores you just saw describe how a random baseline
behaves. To have *test.sh* evaluate your model, follow the instructions in the slide.
