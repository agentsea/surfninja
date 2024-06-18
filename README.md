<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://github.com/agentsea/skillpacks">
    <img src="https://project-logo.png" alt="Logo" width="80">
  </a> -->

  <h1 align="center">SurfNinja</h1>
    <p align="center">
    <img src="logo/SurfNinja-512x512.jpg" alt="SurfSlicer Logo" width="200" style="border-radius: 20px;">
    </p>
  <p align="center">
    A precision based GUI surfer
    <br />
    <a href="https://docs.hub.agentsea.ai/introduction"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/agentsea/surfninja">View Demo</a>
    ·
    <a href="https://github.com/agentsea/surfninja/issues">Report Bug</a>
    ·
    <a href="https://github.com/agentsea/surfninja/issues">Request Feature</a>
  </p>
  <br>
</p>

## Install

```sh
pip install surfkit
```

## Quick Start

Create a tracker

```sh
surfkit create tracker
```

Create a device

```sh
surfkit create device --provider gce --name george
```

Solve a task

```sh
surfkit solve --description "Search for common varieties of french ducks" \
--device george --agent-file ./agent.yaml --runtime docker --kill
```

## Usage

Create an agent

```sh
surfkit create agent -f ./agent.yaml --runtime { process | docker | kube } --name foo
```

List running agents

```sh
surfkit list agents
```

Use the agent to solve a task

```sh
surfkit solve --agent foo --description "Search for french ducks" --device-type desktop
```

Get the agent logs

```sh
surfkit logs --name foo
```

Delete the agent

```sh
surfkit delete agent --name foo
```

## Developing

Install dependencies

```sh
poetry install
```

Create a tracker

```sh
surfkit create tracker
```

Create a device

```sh
surfkit create device --provider gce --name george
```

Solve a task

```sh
surfkit solve --description "Search for common varieties of french ducks" \
--device george --agent-file ./agent.yaml --runtime process --kill
```

## Community

Come join us on [Discord](https://discord.gg/hhaq7XYPS6).

