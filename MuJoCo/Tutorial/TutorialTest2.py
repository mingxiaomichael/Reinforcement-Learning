import mujoco
from mujoco import viewer

# Add joint
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
  <actuator
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

viewer = viewer.launch_passive(model=model, data=data)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 5.0

# simulate and display video
mujoco.mj_resetData(model, data)  # reset state and time
mujoco.mj_forward(model, data)
print(data.time)
# while data.time < duration:
while True:
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, scene_option=scene_option)
    renderer.render()
    print(data.time)
