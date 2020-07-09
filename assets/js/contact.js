
(function ($) {
    "use strict";
    jQuery(document).ready(function ($) {
        $(document).on('submit', '#get_in_touch', function (e) {
            e.preventDefault();
            var firstName = $('#first_name').val();
            var lastName = $('#last_name').val();
            var subject = $('#subject').val();
            var email = $('#email').val();
            var message = $('#message').val();

            if (!firstName) {
                 $('#first_name').removeClass('error');
                 $('#first_name').addClass('error').attr('placeholder', 'First Name');
             }else{
                 $('#first_name').removeClass('error');
             }
           
            if (!lastName) {
                 $('#last_name').removeClass('error');
                 $('#last_name').addClass('error').attr('placeholder', 'Last Name');
             }else{
                 $('#last_name').removeClass('error');
             }
            if (!subject) {
                 $('#subject').removeClass('error');
                 $('#subject').addClass('error').attr('placeholder', 'Subject');
             }else{
                 $('#subject').removeClass('error');
             }
           
            if (!email) {
                 $('#email').removeClass('error');
                 $('#email').addClass('error').attr('placeholder','Enter Your Email')
             }else{
                 $('#email').removeClass('error');
             }

            if (!message) {
                 $('#message').removeClass('error');
                 $('#message').addClass('error').attr('placeholder','Write Your Message')
            }else{
                 $('#message').removeClass('error');
            }
             
            
            if ( email && message && firstName && lastName && subject ) {
             	$.ajax({
	                 type: "POST",
	                 url:'contact.php',
	                 data:{
                         'first_name': firstName,
                         'last_name': lastName,
                         'subject': subject,
                         'email': email,
                         'message': message,
	                 },
	                 success:function(data){
                         $('#get_in_touch').children('.email-success').remove();
                         $('#get_in_touch').prepend('<div class="alert alert-success email-success">' + data + '</div>');
                         $('#first_name').val('');
                         $('#last_name').val('');
                         $('#subject').val('');
                         $('#email').val('');
                         $('#message').val('');
                         $('.email-success').fadeOut(5000);
	                 },
	                 error:function(res){

	                 }
	             });
             }else{
                $('#get_in_touch').children('.email-success').remove();
                $('#get_in_touch').prepend('<div class="alert alert-danger email-success">Somenthing is wrong</div>');
                $('.email-success').fadeOut(5000);
             }
        });
    })

}(jQuery));	
