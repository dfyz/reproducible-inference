FROM alpine:3.23.3

ENV PYTHONDONTWRITEBYTECODE=1

RUN apk add git python3 py3-pyaml py3-pytest libfaketime

# Set the global fake time.
RUN echo '2026-03-14 03:14:15' > /etc/faketimerc

# Patch out printing duration in pytest.
RUN sed \
    's/"""Format the given seconds in a human readable manner to show in the final summary."""/return "due time"/' \
    -i /usr/lib/python3.12/site-packages/_pytest/terminal.py

RUN git clone https://github.com/dfyz/whenwords.git /root/whenwords
# Only leave PROMPT.md and tests.yaml
RUN find /root/whenwords -mindepth 1 -maxdepth 1 ! -name 'PROMPT.md' -exec rm -rf -- {} +
COPY smol_agent.py /root

WORKDIR /root/whenwords
