//
//  CaptureImageView.swift
//  Face Anonymizer
//
//  Created by Jose Bouza on 4/10/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//

import SwiftUI


extension CaptureImageView: UIViewControllerRepresentable {
    func makeUIViewController(context: UIViewControllerRepresentableContext<CaptureImageView>) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController,
                                context: UIViewControllerRepresentableContext<CaptureImageView>) {
        
    }
}
