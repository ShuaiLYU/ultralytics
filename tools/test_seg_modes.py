import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE


# for scale in ['n','s','m','l','x']:
#     print(f"Validating YOLOE-{scale}-seg with text prompt...")
#     # Initialize a YOLOE model
#     model = YOLOE(f"weights/yoloe-26{scale}-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

#     # Set text prompt to detect person and bus. You only need to do this once after you load the model.
#     names = ["a person", "bus"]
#     model.set_classes(names, model.get_text_pe(names))

#     # Run detection on the given image
#     results = model.predict("ultralytics/assets/bus.jpg")

#     # Show results
#     results[0].save(f"runs/output_{scale}.jpg")  # save results to output.jpg


for scale in ['n','s','m','l','x']:
    import numpy as np

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    # Initialize a YOLOE model
    model = YOLOE(f"weights/yoloe-26{scale}-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes


    # Define visual prompts using bounding boxes and their corresponding class IDs.
    # Each box highlights an example of the object you want the model to detect.
    visual_prompts = dict(
        bboxes=np.array(
            [
                [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
                [120, 425, 160, 445],  # Box enclosing glasses
            ],
        ),
        cls=np.array(
            [
                0,  # ID to be assigned for person
                1,  # ID to be assigned for glassses
            ]
        ),
    )

    # Run inference on an image, using the provided visual prompts as guidance
    results = model.predict(
        "ultralytics/assets/bus.jpg",
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=0.25,
    )

    #     # Show results
    print(f"saveing runs/output_{scale}-vp.jpg")
    results[0].save(f"runs/output_{scale}-vp.jpg")  # save results to output.jpg
