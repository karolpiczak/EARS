function _base64ToArrayBuffer(base64) {
    var binary_string =  window.atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array( len );
    for (var i = 0; i < len; i++)        {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes.buffer;
}

// var wavesurfer = WaveSurfer.create({
//     container: '#waveform'
// });
//
// wavesurfer.on('ready', function () {
//     wavesurfer.play();
// });

// wavesurfer.loadArrayBuffer(_base64ToArrayBuffer(audio_base64));
