/* Scores and examples transcribed from the accompanying paper, Tables 1-5. */
window.SPATIAL_DATA = {
  main: [
    {
      name: "GPT-5.5",
      group: "proprietary",
      values: [55.4, 39.2, 45.5, 63.9, 23.9, 48.1, 56.0, 53.4],
    },
    {
      name: "GPT-4o",
      group: "proprietary",
      values: [37.0, 31.5, 34.0, 36.5, 23.4, 38.2, 48.2, 39.2],
    },
    {
      name: "Gemini-2.5-Pro",
      group: "proprietary",
      values: [61.1, 45.9, 51.5, 52.2, 8.0, 42.4, 58.3, 51.1],
    },
    {
      name: "Qwen2.5-VL-3B",
      group: "base",
      values: [33.8, 27.3, 27.0, 33.2, 15.3, 38.6, 36.6, 33.9],
    },
    {
      name: "Qwen2.5-VL-7B",
      group: "base",
      values: [36.6, 30.4, 37.9, 29.3, 8.6, 39.0, 34.0, 35.1],
    },
    {
      name: "Qwen3-VL-4B",
      group: "base",
      values: [52.4, 34.0, 52.8, 27.2, 25.9, 44.8, 64.2, 47.3],
    },
    {
      name: "Qwen3-VL-8B",
      group: "base",
      values: [53.5, 32.0, 55.7, 29.4, 24.2, 43.3, 65.1, 48.4],
    },
    {
      name: "InternVL3-2B",
      group: "open",
      values: [32.2, 32.9, 32.9, 37.5, 21.4, 32.5, 37.6, 35.1],
    },
    {
      name: "InternVL3-8B",
      group: "open",
      values: [48.0, 26.3, 42.1, 41.5, 22.4, 43.1, 45.5, 43.1],
    },
    {
      name: "SpaceR-7B",
      group: "spatial",
      values: [39.7, 31.4, 44.5, 29.9, 12.7, 43.7, 56.9, 43.8],
    },
    {
      name: "Video-R1-7B",
      group: "spatial",
      values: [36.8, 31.4, 33.4, 30.9, 5.4, 36.2, 40.7, 35.3],
    },
    {
      name: "VILASR-7B",
      group: "spatial",
      values: [45.1, 29.9, 44.6, 35.1, 15.7, 46.2, 57.6, 45.9],
    },
    {
      name: "Spatial-MLLM-4B",
      group: "spatial",
      values: [40.4, 33.0, 46.3, 32.1, 16.9, 38.0, 58.9, 43.8],
    },
    {
      name: "SpatialLadder-3B",
      group: "spatial",
      values: [45.6, 27.3, 44.8, 43.5, 29.7, 46.2, 70.9, 51.4],
    },
    {
      name: "Spatial-Interactor-3B",
      group: "ours",
      base: "Qwen2.5-VL-3B",
      values: [53.0, 33.0, 50.4, 59.3, 32.4, 48.3, 64.5, 55.6],
    },
    {
      name: "Spatial-Interactor-4B",
      group: "ours",
      base: "Qwen3-VL-4B",
      values: [60.7, 36.6, 57.2, 80.3, 31.0, 47.1, 70.3, 63.7],
    },
    {
      name: "Spatial-Interactor-7B",
      group: "ours",
      base: "Qwen2.5-VL-7B",
      values: [50.6, 34.0, 49.0, 74.3, 36.6, 49.5, 67.6, 60.1],
    },
    {
      name: "Spatial-Interactor-8B",
      group: "ours",
      base: "Qwen3-VL-8B",
      values: [62.7, 39.7, 60.0, 88.4, 33.2, 47.1, 68.0, 65.9],
    },
  ],
  generalization: [
    {
      name: "Qwen2.5-VL-3B",
      group: "base",
      values: [29.0, 35.6, 56.7, 56.3, 44.4],
    },
    { name: "3B + SFT", group: "sft", values: [29.2, 43.2, 61.0, 56.8, 47.6] },
    {
      name: "Spatial-Interactor-3B",
      group: "ours",
      values: [29.9, 44.3, 62.0, 58.7, 48.7],
    },
    {
      name: "Qwen2.5-VL-7B",
      group: "base",
      values: [27.7, 36.3, 54.7, 57.5, 44.1],
    },
    { name: "7B + SFT", group: "sft", values: [30.4, 41.3, 71.3, 61.3, 51.1] },
    {
      name: "Spatial-Interactor-7B",
      group: "ours",
      values: [31.2, 42.4, 74.0, 62.5, 52.5],
    },
    {
      name: "Qwen3-VL-4B",
      group: "base",
      values: [29.0, 35.2, 60.7, 60.9, 46.5],
    },
    { name: "4B + SFT", group: "sft", values: [30.3, 43.2, 72.7, 62.7, 52.2] },
    {
      name: "Spatial-Interactor-4B",
      group: "ours",
      values: [31.0, 43.5, 72.7, 65.7, 53.2],
    },
    {
      name: "Qwen3-VL-8B",
      group: "base",
      values: [30.8, 40.3, 52.7, 43.6, 41.9],
    },
    { name: "8B + SFT", group: "sft", values: [31.0, 43.8, 72.7, 58.3, 51.5] },
    {
      name: "Spatial-Interactor-8B",
      group: "ours",
      values: [32.2, 44.3, 74.0, 59.5, 52.5],
    },
  ],
  ablation: [
    {
      name: "Base",
      group: "base",
      values: [36.6, 30.4, 37.9, 29.3, 8.6, 39.0, 34.0, 35.1],
    },
    {
      name: "External-data SFT",
      group: "sft",
      values: [48.8, 30.4, 46.2, 70.8, 30.1, 45.3, 61.6, 56.0],
    },
    {
      name: "Full SFT",
      group: "sft",
      values: [49.9, 29.4, 48.2, 72.2, 32.6, 47.5, 65.5, 58.4],
    },
    {
      name: "Full SFT + GRPO",
      group: "grpo",
      values: [50.3, 28.9, 48.1, 73.1, 33.5, 48.6, 67.0, 59.2],
    },
    {
      name: "Full SFT + OPD",
      group: "ours",
      values: [50.6, 34.0, 49.0, 74.3, 36.6, 49.5, 67.6, 60.1],
    },
  ],
  interaction: {
    walker: {
      paperTable: 4,
      title: "WalkerBench Standard-100",
      columns: [
        "Angle",
        "Distance",
        "Height",
        "Navigation",
        "Visibility",
        "Overall",
        "Steps",
      ],
      rows: [
        {
          name: "Base",
          group: "base",
          values: [5.0, 0.0, 25.0, 5.0, 0.0, 7.0, 17.71],
        },
        {
          name: "Spatial-Interactor",
          group: "ours",
          values: [20.0, 0.0, 40.0, 10.0, 0.0, 14.0, 6.52],
        },
      ],
    },
    esi: {
      paperTable: 5,
      title: "ESI-Bench",
      columns: [
        "Rigid",
        "Stability",
        "View",
        "Occlusion",
        "Distance",
        "Agent obs.",
        "Unobserved",
        "Action order",
        "Overall",
        "Steps",
      ],
      rows: [
        {
          name: "Base",
          group: "base",
          values: [33.3, 26.7, 56.7, 23.3, 50.0, 20.0, 26.7, 26.7, 32.9, 12.34],
        },
        {
          name: "Spatial-Interactor",
          group: "ours",
          values: [40.0, 36.7, 63.3, 33.3, 56.7, 33.3, 33.3, 33.3, 41.2, 10.55],
        },
      ],
    },
  },
  levels: {
    l1: {
      title: "Passive world-state transitions",
      count: "15,109",
      description:
        "The world changes; the viewpoint stays approximately fixed. Learn object motion, state changes, and single- or multi-step operations.",
      tasks: [
        "State changes",
        "Object motion",
        "Operation magnitude",
        "Operation order",
        "Multi-step operation",
      ],
      cases: ["E02", "E03", "E09", "E28"],
    },
    l2: {
      title: "Active self-state transitions",
      count: "69,487",
      description:
        "The observer moves through a stable environment. Explain visual changes through ego-motion and reason across viewpoints.",
      tasks: [
        "Camera motion",
        "Motion comparison",
        "Temporal ordering",
        "Parallax & depth",
        "Perspective taking",
      ],
      cases: ["E12", "E13", "E22", "E23"],
    },
    l3: {
      title: "Long-horizon integration",
      count: "22,922",
      description:
        "Integrate successive camera-motion transitions across a complete trajectory to recover global paths and identify key locations.",
      tasks: [
        "Path length",
        "Displacement",
        "Path shape",
        "Reverse paths",
        "Turns & revisits",
      ],
      cases: ["E32", "E29", "E33", "E34"],
    },
  },
  cases: {
    E02: {
      title: "Open / close",
      source: "ProcTHOR",
      domain: "Simulated",
      question: "What state change occurred between the two views?",
      options: [
        "An object was rotated",
        "An object was turned on or off",
        "An object was opened or closed",
        "An object was removed from the scene",
      ],
      answer: "C",
    },
    E03: {
      title: "Object rotation",
      source: "ProcTHOR",
      domain: "Simulated",
      question: "What state change occurred between the two views?",
      options: [
        "An object was turned on or off",
        "An object was removed from the scene",
        "An object was opened or closed",
        "An object was rotated",
      ],
      answer: "D",
    },
    E09: {
      title: "Operation magnitude",
      source: "BridgeData V2",
      domain: "Real robot",
      question:
        "The end effector moves right between the two views. Approximately how far does it move?",
      options: ["About 5 cm", "About 10 cm", "About 15 cm", "About 20 cm"],
      answer: "B",
    },
    E28: {
      title: "Multi-step operation",
      source: "BridgeData V2",
      domain: "Real robot",
      question:
        "For the instruction 'put pepper in pan,' which sequence of horizontal movements best matches the video?",
      options: [
        "First moves right about 30 cm, then moves forward about 15 cm",
        "First moves right about 15 cm, then moves forward about 30 cm",
        "First moves forward about 30 cm, then moves right about 15 cm",
        "First moves right about 20 cm, then moves forward about 25 cm",
      ],
      answer: "A",
    },
    E12: {
      title: "Camera rotation",
      source: "ProcTHOR",
      domain: "Simulated",
      question: "Which camera motion connects the two views?",
      options: [
        "Move left by 0.6 meters",
        "Rotate left by 60 degrees",
        "Move left by 0.8 meters",
        "Rotate left by 45 degrees",
      ],
      answer: "B",
    },
    E13: {
      title: "Composed camera motion",
      source: "AI2-THOR",
      domain: "Simulated",
      question: "Which two camera motions connect the views?",
      options: [
        "Move forward by 0.6 meters, then rotate right by 60 degrees",
        "Rotate right by 60 degrees, then move forward by 0.6 meters",
        "Rotate right by 60 degrees, then move forward by 0.8 meters",
        "Move forward by 0.8 meters, then rotate right by 75 degrees",
      ],
      answer: "A",
    },
    E22: {
      title: "Parallax depth",
      source: "ProcTHOR",
      domain: "Simulated",
      question:
        "Using parallax between the two views, which is closer to the camera: the vase or the chair?",
      options: ["Cannot tell", "Vase", "Same depth", "Chair"],
      answer: "B",
    },
    E23: {
      title: "Imagined perspective",
      source: "AI2-THOR",
      domain: "Simulated",
      question:
        "Imagine standing at the mirror and facing into the scene. Where is the sink relative to you?",
      options: ["Back-left", "Front-right", "Front-left", "Back-right"],
      answer: "B",
    },
    E29: {
      title: "Reverse-path planning",
      source: "RoomTour3D",
      domain: "Real camera",
      question: "Which reverse path returns to the starting location?",
      options: [
        "Turn around, move forward, then turn right and move forward",
        "Turn around, move forward, then turn left and move forward",
        "Turn around, move forward",
        "Move straight forward",
      ],
      answer: "A",
    },
    E32: {
      title: "Path shape",
      source: "RoomTour3D",
      domain: "Real camera",
      question: "Which path shape describes the complete video?",
      options: [
        "Moves forward, turns left, then continues forward",
        "Moves forward, turns left, continues forward, then turns left and continues forward",
        "Moves forward, turns right, then continues forward",
        "Moves mostly straight forward",
      ],
      answer: "D",
    },
    E33: {
      title: "Metric trajectory",
      source: "RoomTour3D",
      domain: "Real camera",
      question: "What are the displacement, turn, and total path length?",
      fields: [
        {
          label: "Displacement",
          options: ["6.5 m", "7.0 m", "6.0 m", "5.5 m"],
          answer: "A",
        },
        {
          label: "Turn",
          options: [
            "Right 30 degrees",
            "Right 90 degrees",
            "Right 60 degrees",
            "0 degrees (straight ahead)",
          ],
          answer: "A",
        },
        {
          label: "Path length",
          options: ["6.5 m", "8.5 m", "9.5 m", "7.5 m"],
          answer: "B",
        },
      ],
      answer: "A / A / B",
    },
    E34: {
      title: "Turn localization",
      source: "RoomTour3D",
      domain: "Real camera",
      question: "In which quarter does the camera turn while moving?",
      options: [
        "The final quarter, around frames 25-32",
        "The first quarter, around frames 1-8",
        "The third quarter, around frames 17-24",
        "The second quarter, around frames 9-16",
      ],
      answer: "D",
    },
  },
};
