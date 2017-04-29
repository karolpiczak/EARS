function _base64ToArrayBuffer(base64) {
    var binary_string =  window.atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array( len );
    for (var i = 0; i < len; i++)        {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes.buffer;
}

var AudioContext = AudioContext || webkitAudioContext || mozAudioContext;
var audio_context = new AudioContext();
var next_chunk_time = 0;
var is_muted = false;