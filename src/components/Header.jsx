import { NavLink } from 'react-router-dom'
import logo from '../assets/logo_abs.png'


const Header=()=>{
    return(
        <div className='header'>
          <nav className='navBar'>
           <NavLink to="/home">Accueil</NavLink>
           <NavLink to="/services">Services</NavLink>
           <NavLink to="/contacts">Contacts</NavLink>
           <NavLink to="/about">A Propos</NavLink>
          </nav>
          <div className="logo">
            <img src={logo} className="logo_abs" alt='logo ABShopp'/>
            <span className="shoppName">ABShopp</span>
          </div>
          <input className="search" placeholder="rechercher"/>
        </div>
    )
}

export default Header

