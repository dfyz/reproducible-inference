FROM alpine:3.23.3

RUN apk add git python3 py3-pyaml py3-pytest libfaketime

# Set the global fake time.
RUN echo '2026-03-14 03:14:15' > /etc/faketimerc

# Patch out printing duration in pytest.
RUN sed \
    's/"""Format the given seconds in a human readable manner to show in the final summary."""/return "due time"/' \
    -i /usr/lib/python3.12/site-packages/_pytest/terminal.py

RUN git clone https://github.com/dbreunig/whenwords.git /root/whenwords
WORKDIR /root
COPY smol_agent.py .
COPY whenwords.txt .
RUN mkdir -p results/a results/b

WORKDIR /root/whenwords

