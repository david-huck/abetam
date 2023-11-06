FROM condaforge/mambaforge:4.9.2-5 as conda

WORKDIR /app

# if using 
# # install current environment or use cached one
# RUN apt-get -y update

# RUN apt install gnupg -y
# # # Adding trusting keys to apt for repositories
# RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -

# # # Adding Google Chrome to the repositories
# RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list'

# RUN apt update && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y google-chrome-stable

# # Magic happens
# RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# RUN dpkg -i google-chrome-stable_current_amd64.deb


COPY env.yml env.yml

RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create --file env.yml  && echo 4
# COPY . /pkg


# FROM selenium/standalone-chrome:latest
# WORKDIR /app
# COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yaml
# RUN micromamba install -y -n base -f /tmp/env.yaml && \
#     micromamba clean --all --yes
# ARG MAMBA_DOCKERFILE_ACTIVATE=1
# ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
RUN echo "source activate tau" > ~/.bashrc
ENV PATH /opt/conda/envs/tau/bin:$PATH

RUN pip install pyogrio
COPY . .
ENV PORT=8501

EXPOSE ${PORT}

CMD [ "streamlit", "run", "abm.py" ]