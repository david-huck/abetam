FROM condaforge/mambaforge:4.9.2-5 as conda

WORKDIR /app

COPY env.yml env.yml

RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create --file env.yml  && echo 4

ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
RUN echo "source activate tau" > ~/.bashrc
ENV PATH /opt/conda/envs/tau/bin:$PATH

COPY . .
ENV PORT=8501

EXPOSE ${PORT}

CMD [ "streamlit", "run", "app.py" ]
