FROM alpine:3.23.3

RUN apk add bash npm fd ripgrep git python3 py3-pyaml py3-pytest libfaketime

# Set the global fake time.
RUN echo '2026-03-14 03:14:15' > /etc/faketimerc

# Patch out printing duration in pytest.
RUN sed \
    's/"""Format the given seconds in a human readable manner to show in the final summary."""/return "due time"/' \
    -i /usr/lib/python3.12/site-packages/_pytest/terminal.py

RUN npm install -g @mariozechner/pi-coding-agent

RUN git clone https://github.com/dbreunig/whenwords.git /root/whenwords
WORKDIR /root/whenwords

#COPY models.json /root/.pi/agent/models.json
COPY auth.json /root/.pi/agent/auth.json
COPY settings.json /root/.pi/agent/settings.json

#ENV LD_PRELOAD=/usr/lib/faketime/libfaketime.so.1
#ENV FAKETIME_DONT_FAKE_MONOTONIC=1
CMD ["pi", "--verbose"]
