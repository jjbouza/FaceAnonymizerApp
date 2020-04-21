//
//  Coordinator.swift
//  Face Anonymizer
//
//  Created by Jose Bouza on 4/10/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//

import SwiftUI
class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
  @Binding var isCoordinatorShown: Bool
  @Binding var imageInCoordinator: Image?
    @Binding var im_size : CGSize?
    init(isShown: Binding<Bool>, image: Binding<Image?>, size: Binding<CGSize?>) {
    _isCoordinatorShown = isShown
    _imageInCoordinator = image
    _im_size = size
  }
  func imagePickerController(_ picker: UIImagePickerController,
                didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
     guard let unwrapImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else { return }
    
     let network_process = NetworkProcessing()
     let image_out_ = network_process.predict(image: unwrapImage)
    let image_out = image_out_.resized(toWidth: 400)
     imageInCoordinator = Image(uiImage: image_out)
     isCoordinatorShown = false
     let __im_size = CGSize(width: image_out.cgImage!.width, height: image_out.cgImage!.height)
     im_size = __im_size
    
  }
  func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
     isCoordinatorShown = false
  }
}
