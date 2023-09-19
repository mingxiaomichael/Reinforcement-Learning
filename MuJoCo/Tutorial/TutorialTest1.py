import mujoco
from mujoco import viewer


xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)

viewer = viewer.launch_passive(model=model, data=data)

duration = 5.0

# simulate and display video
mujoco.mj_resetData(model, data)  # reset state and time
print(data.time)
while data.time < duration:
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    renderer.render()
    print(data.time)

