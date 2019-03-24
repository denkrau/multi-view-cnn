"""Generates views from an .off object using the Blender Python API.

Run inside Blender Python Console with

file = "D:\\Dropbox\\Studium\\Master\\Masterthesis\\Code\\generate_views.py"
exec(compile(open(file).read(), "file", 'exec'))

"""

import os
import collections
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

def is_face_visible(face):
	faces = obj.data.polygons
	#get coordinates of face vertices and move them slighty in direction of face center for unique vertex distinction
	vertices_mesh = obj.data.vertices
	v1 = vertices_mesh[face.vertices[0]].co + 0.05*(face.center - vertices_mesh[face.vertices[0]].co)
	v2 = vertices_mesh[face.vertices[1]].co + 0.05*(face.center - vertices_mesh[face.vertices[1]].co)
	v3 = vertices_mesh[face.vertices[2]].co + 0.05*(face.center - vertices_mesh[face.vertices[2]].co)
	face_coords = [v1, v2, v3]

	#check by ray cast if face vertices are seen by camera
	#if true a visible face is supposed
	#this is done for every view
	cam.location = (0, 0, 0)
	cam.rotation_euler[0] = 60*3.141/180
	is_hidden = False
	is_visible = False
	for i in range(n_views):
		cam.rotation_euler[2] = i*camera_steps*3.141/180
		bpy.ops.view3d.camera_to_view_selected()
		#apply padding to render by moving camera away along its z-axis
		cam.location = cam.matrix_world * Vector((0,0,6))
		number_of_vertex_misses = 0
		for c in face_coords:
			#result = [is_hit, hit_location, face_normal, face_id]
			result = obj.ray_cast(obj.matrix_world.inverted()*cam.location, c-obj.matrix_world.inverted()*cam.location)
			if result[0]:
				if faces[result[3]] == face:
					#face is visible so interrupt further checking
					is_visible = True
					break
			#if face is not visible increment counter
			number_of_vertex_misses += 1
		#if whole face is not vissible in current view increment counter
		if number_of_vertex_misses == len(face_coords):
			is_hidden = True
		#if face was at least once seen and at least once hidden in different views return face
		if is_visible and is_hidden:
			return True

	return False


def get_optimal_face(cam, obj, colors_per_object=1):
	"""Iterates over all mesh faces to find one that has a decent size and is seen by the camera

		Args:
			scene: blender scene object
			cam: camera object
			obj: mesh object

		Returns:
			face: optimal face object
	"""
	idx = 0
	#get mean over all faces
	faces = obj.data.polygons
	faces_area_mean = np.mean([i.area for i in faces])
	faces_area_max = np.max([i.area for i in faces])

	#get first face whose area is larger than mean
	possible_faces = []
	face = None
	for i, f in enumerate(faces):
		if f.area > (faces_area_mean + faces_area_max) * 0.3:
			possible_faces.append(f)
			face = f

	#sort by center distance to face in descending order
	possible_faces = sorted(possible_faces, key=lambda x: np.linalg.norm(x.center-face.center), reverse=True)
	faces_to_color = []

	for face in possible_faces:
		if is_face_visible(face):
			faces_to_color.append(get_identical_faces(face))
			if colors_per_object > 1:
				possible_faces = sorted(possible_faces, key=lambda x: np.linalg.norm(x.center-face.center), reverse=True)
				for face2 in possible_faces[:-1]:
					if is_face_visible(face2):
						faces_to_color.append(get_identical_faces(face2))
						return faces_to_color
			else:
				return faces_to_color
	return None

def get_identical_faces(face):
	"""Finds faces with identical vertice coordinates to given face.

	Args:
		face: face to find identical faces to

	Returns:
		identical_faces: list of identical face objects
	"""	
	identical_faces = [face]
	vertices = obj.data.vertices
	faces = obj.data.polygons
	face_idx = list(faces).index(face)
	face_coords = []
	for v in face.vertices:
		face_coords.append(vertices[v].co)
	for f in faces[face_idx+1:]:
		iter_coords = []
		sum_matches = 0
		for v in f.vertices:
			if not vertices[v].co in face_coords:
				break
			sum_matches += 1
		if sum_matches == 3:
			identical_faces.append(f)
			
	return identical_faces

# define views and camera position
n_views = 12
camera_steps = 360/n_views
n_train_objects = 80
n_test_objects = 20
colors_per_object = 1
n_materials = 2

# define paths
input = "D:\\Downloads\\Web\\ModelNet10"
output = "D:\\Downloads\\Web\\ModelNet10views"

# define each material and color
materials = collections.OrderedDict([("green", (0,255,0)), ("red", (255,0,0)), ("orange", (255,165,0)), ("blue", (0,0,255))])

#delete default cube
if "Cube" in bpy.data.objects:
	bpy.ops.object.select_all(action='DESELECT')
	bpy.data.objects["Cube"].select = True
	bpy.ops.object.delete()

#get context objects
cam = bpy.data.objects["Camera"]
scene = bpy.data.scenes["Scene"]

#modify environment
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224
scene.render.resolution_percentage = 100
scene.render.antialiasing_samples = "16"
scene.render.use_full_sample = True
bpy.data.cameras["Camera"].clip_end = 9999.9
cam.lock_rotation[0] = True
bpy.data.lamps["Lamp"].type = "HEMI"
bpy.data.objects['Lamp'].location = (0, 0, 90)
bpy.data.objects['Lamp'].rotation_euler = (0, 0, 0)

mats = register_materials(materials)

#import mesh and render it
for root, dirs, files in os.walk(input):
	n_objects = 0
	for file in sorted(files):
		if os.path.splitext(file)[1] == ".off":
			path, type = os.path.split(root)
			category = os.path.split(path)[1]
			filename, ext = os.path.splitext(file)
			bpy.ops.import_mesh.off(filepath=os.path.join(root, file))
			bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
			obj = bpy.data.objects[os.path.splitext(file)[0]]
			register_object_materials(obj, mats)
			faces_to_color = get_optimal_face(cam, obj, colors_per_object=colors_per_object)
			if faces_to_color is not None:
				n_objects += 1
				if type == "train":
					if n_objects > n_train_objects:
						# delete current mesh
						bpy.ops.object.select_all(action='DESELECT')
						obj.select = True
						bpy.ops.object.delete()
						break
				elif type == "test":
					if n_objects > n_test_objects:
						# delete current mesh
						bpy.ops.object.select_all(action='DESELECT')
						obj.select = True
						bpy.ops.object.delete()
						break
				for j in range(n_materials):
					for f, faces in enumerate(faces_to_color):
						for face in faces:
							face.material_index = j*len(faces_to_color)+1+f
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