"""Generates views from an .off object using the Blender Python API.

Run inside Blender Python Console with

file = "D:\\Dropbox\\Studium\\Master\\Masterthesis\\Code\\generate_views.py"
exec(compile(open(file).read(), "file", 'exec'))
"""

import os
import numpy as np
import bpy

def register_materials(materials):
	"""Adds materials to scene.

	Args:
		materials: Dict of material names and colors
	Returns:
		mats: List of all created material objects
	"""
	mats = []
	for k, v in materials.items():
		mat = bpy.data.materials.new(k)
		mat.diffuse_color = v
		mat.transparency_method = "Z_TRANSPARENCY"
		mats.append(mat)
	return mats

def register_object_materials(obj, mats):
	"""Appends scene materials to given object.

	Args:
		obj: Mesh object
		mats: List of scene material objects
	"""
	# apply default material to object
	bpy.data.materials["Material"].use_transparency = True
	bpy.data.materials["Material"].transparency_method = "Z_TRANSPARENCY"
	obj.active_material = bpy.data.materials["Material"]

	# add new materials to object's materials
	for i in mats:
		obj.data.materials.append(i)

def color_faces(obj, faces, area_mean, mat_id):
	"""Applies material of scene to faces of given object.
	
	Args:
		obj: Mesh object
		mat_id: Id of material to be applied
	"""
	for f in faces:
		if f.area > area_mean:
			f.material_index = mat_id
			break

# define views and camera position
n_views = 12
camera_steps = 360/n_views

# define paths
input = "D:\\Downloads\\Web\\ModelNet10"
output = "D:\\Downloads\\Web\\ModelNet10views"

# define each material and color
materials = {"green": (0,255,0)}

#delete default cube
if "Cube" in bpy.data.objects:
	bpy.ops.object.select_all(action='DESELECT')
	bpy.data.objects["Cube"].select = True
	bpy.ops.object.delete()

#modify environment
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.data.scenes['Scene'].render.antialiasing_samples = "16"
bpy.data.scenes['Scene'].render.use_full_sample = True
bpy.data.cameras["Camera"].clip_end = 9999.9
bpy.data.objects["Camera"].lock_rotation[0] = True
bpy.data.lamps["Lamp"].type = "HEMI"
bpy.data.objects['Lamp'].location = (0, 0, 90)
bpy.data.objects['Lamp'].rotation_euler = (0, 0, 0)

mats = register_materials(materials)

#import mesh and render it
for root, dirs, files in os.walk(input):
	amount = len(files)
	for i, file in enumerate(sorted(files)):
		if file.endswith(".off"):
			print("Processing file", i, "of", amount, end="\r")
			path, type = os.path.split(root)
			category = os.path.split(path)[1]
			filename, ext = os.path.splitext(file)
			bpy.ops.import_mesh.off(filepath=os.path.join(root, file))
			bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
			obj = bpy.data.objects[os.path.splitext(file)[0]]
			faces = obj.data.polygons
			faces_area_mean = np.mean([i.area for i in faces])
			register_object_materials(obj, mats)
			for j in range(len(materials)+1):
				if j > 0:
					color_faces(obj, faces, faces_area_mean, j)
				# position camera for first view
				bpy.data.objects["Camera"].location = (0, 0, 0)
				bpy.data.objects["Camera"].rotation_euler[0] = 60*3.141/180
				for k in range(n_views):
					bpy.data.objects["Camera"].rotation_euler[2] = k*camera_steps*3.141/180
					bpy.ops.view3d.camera_to_view_selected()
					#apply padding to render by moving camera away along its z-axis
					bpy.data.objects["Camera"].location = bpy.data.objects["Camera"].matrix_local * Vector((0,0,6))
					bpy.context.scene.render.filepath = os.path.join(output, category, type, filename + "_" + str(j) + "_" + str(k).zfill(3) + ".png")
					bpy.ops.render.render(write_still=True)
			# delete current mesh
			bpy.ops.object.select_all(action='DESELECT')
			obj.select = True
			bpy.ops.object.delete()